from copy import deepcopy
import threading

import numpy as np
import pytest

from arena_agents import ArenaAdapter, Decision
from game import Game2048
from web_game import (
    AgentFailure,
    GameService,
    InvalidRequest,
    StateConflict,
)


class FirstLegalAdapter(ArenaAdapter):
    name = "Expectimax"

    def decide(self, board, legal_actions):
        action = legal_actions[0] if legal_actions else None
        return Decision(action, list(legal_actions), latency_seconds=0.001)


class WaitingAdapter(FirstLegalAdapter):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def decide(self, board, legal_actions):
        self.started.set()
        assert self.release.wait(timeout=3)
        return super().decide(board, legal_actions)


class BrokenAdapter(FirstLegalAdapter):
    def decide(self, board, legal_actions):
        raise RuntimeError("synthetic agent failure")


def service(adapter=None):
    return GameService(adapters={"Expectimax": adapter or FirstLegalAdapter()})


def test_human_move_changes_board_and_undo_restores_rng():
    app = service()
    state = app.new_game(seed=2)
    before = np.asarray(state["board"])
    rng_before = deepcopy(app._sessions[state["id"]].env.rng.bit_generator.state)
    action = next(name for name in state["valid_actions"] if not np.array_equal(
        Game2048._move_board(before, name)[0], before
    ))

    after = app.move(state["id"], action, state["version"])
    assert not np.array_equal(after["board"], before)
    assert after["move_count"] == 1
    restored = app.undo_move(state["id"])
    assert restored["board"] == before.tolist()
    assert restored["move_count"] == 0
    assert app._sessions[state["id"]].env.rng.bit_generator.state == rng_before


def test_ai_can_start_from_partially_played_game_and_human_can_resume():
    app = service()
    state = app.new_game(seed=3)
    state = app.move(state["id"], state["valid_actions"][0], state["version"])
    state = app.set_control(state["id"], "ai")
    state = app.ai_move(state["id"], state["version"])
    assert state["move_count"] == 2

    state = app.set_control(state["id"], "human")
    assert state["control"] == "human"
    state = app.move(state["id"], state["valid_actions"][0], state["version"])
    assert state["move_count"] == 3

    state = app.set_control(state["id"], "ai")
    state = app.ai_move(state["id"], state["version"])
    assert state["move_count"] == 4


def test_take_control_cancels_a_pending_decision_without_stale_move():
    adapter = WaitingAdapter()
    app = service(adapter)
    state = app.new_game(seed=4)
    state = app.set_control(state["id"], "ai")
    before_board = state["board"]
    errors = []

    def request_move():
        try:
            app.ai_move(state["id"], state["version"])
        except Exception as exc:  # collected from the worker thread
            errors.append(exc)

    worker = threading.Thread(target=request_move)
    worker.start()
    assert adapter.started.wait(timeout=2)
    stopped = app.set_control(state["id"], "human")
    adapter.release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], StateConflict)
    assert stopped["board"] == before_board
    assert app.get_game(state["id"])["move_count"] == 0
    assert app.get_game(state["id"])["control"] == "human"


def test_sessions_do_not_share_boards():
    app = service()
    first = app.new_game(seed=10)
    second = app.new_game(seed=11)
    app.move(first["id"], first["valid_actions"][0], first["version"])
    assert app.get_game(second["id"])["move_count"] == 0
    assert app.get_game(second["id"])["board"] == second["board"]


def test_invalid_and_stale_human_moves_are_rejected():
    app = service()
    state = app.new_game(seed=12)
    app._sessions[state["id"]].env.board = np.array(
        [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    state = app.get_game(state["id"])
    invalid = "left"
    with pytest.raises(InvalidRequest):
        app.move(state["id"], invalid, state["version"])
    with pytest.raises(StateConflict):
        app.move(state["id"], state["valid_actions"][0], state["version"] + 1)
    assert app.get_game(state["id"])["move_count"] == 0


def test_game_over_blocks_ai_and_reports_finished():
    app = service()
    state = app.new_game(seed=13)
    session = app._sessions[state["id"]]
    session.env.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    session.env.done = True
    state = app.get_game(state["id"])
    assert state["status"] == "finished"
    with pytest.raises(StateConflict):
        app.set_control(state["id"], "ai")


def test_agent_error_does_not_change_game_state():
    app = service(BrokenAdapter())
    state = app.new_game(seed=14)
    state = app.set_control(state["id"], "ai")
    before = app.get_game(state["id"])
    with pytest.raises(AgentFailure):
        app.ai_move(state["id"], state["version"])
    after = app.get_game(state["id"])
    assert after["board"] == before["board"]
    assert after["score"] == before["score"]
    assert after["move_count"] == before["move_count"]
    assert after["control"] == "human"
    assert after["status"] == "error"


def test_legacy_strategy_v1_is_disabled_when_mapping_is_unverified():
    from web_game import AGENT_DESCRIPTIONS

    assert "unverified" in AGENT_DESCRIPTIONS["Strategy V1 DQN"]
    app = service()
    option = next(item for item in app.get_game(app.new_game()["id"])["agents"] if item["name"] == "Strategy V1 DQN")
    assert option["available"] is False


def test_missing_dqn_checkpoints_are_disabled_without_fallback(monkeypatch, tmp_path):
    from web_game import build_web_adapters

    monkeypatch.setenv("DQN_SHAPED_CHECKPOINT", str(tmp_path / "missing-shaped.pt"))
    monkeypatch.setenv("DQN_RAW_SCORE_CHECKPOINT", str(tmp_path / "missing-raw.pt"))
    adapters = build_web_adapters()
    for name in ("Original Dueling Double DQN", "Raw-Score Dueling DQN"):
        assert adapters[name].metadata()["loading_status"] == "unavailable"
        assert name in adapters[name].metadata()["agent_id"]


def test_oversized_checkpoint_is_disabled_before_loading(monkeypatch, tmp_path):
    from web_game import build_web_adapters

    checkpoint = tmp_path / "oversized.pt"
    checkpoint.write_bytes(b"0" * (1024 * 1024 + 1))
    monkeypatch.setenv("DQN_SHAPED_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DQN_RAW_SCORE_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("WEB_MAX_CHECKPOINT_MB", "1")
    adapter = build_web_adapters()["Original Dueling Double DQN"]
    assert adapter.metadata()["loading_status"] == "unavailable"
    assert "exceeds WEB_MAX_CHECKPOINT_MB" in adapter.metadata()["error"]


def test_canonical_benchmarks_are_loaded_from_verified_artifacts():
    from pathlib import Path

    results = GameService(
        adapters={"Expectimax": FirstLegalAdapter()},
        benchmark_path=Path(__file__).resolve().parents[1] / "results/canonical",
    ).benchmarks()
    assert len(results["rows"]) == 4
    expectimax = next(row for row in results["rows"] if row["agent"] == "Expectimax depth 2")
    assert expectimax["mean_score"] == 14450.44
    assert expectimax["reached_2048_pct"] == 15.0
