import numpy as np
import pytest

from arena_agents import (
    ACTION_NAMES,
    ArenaState,
    Decision,
    build_adapters,
)
from game import Game2048


def test_action_mapping_matches_game():
    assert ACTION_NAMES == ("up", "right", "down", "left")
    assert Game2048._move_board(np.array([[2, 0, 0, 0]] * 4), 0)[0].shape == (4, 4)


def test_all_available_adapters_make_legal_decisions():
    adapters = build_adapters(depth=1)
    board = Game2048(seed=5).board
    legal = Game2048(seed=5).valid_actions()
    for name, adapter in adapters.items():
        if adapter.metadata().get("loading_status") == "unavailable":
            continue
        decision = adapter.decide(board, legal)
        assert decision.action in legal, name
        assert decision.action_name in ACTION_NAMES
        assert decision.diagnostics is not None


def test_strategy_v1_mapping_is_explicitly_unverified():
    metadata = build_adapters(depth=1)["Strategy V1 DQN"].metadata()
    assert metadata["action_mapping_status"] == "assumed_unverified"


def test_missing_checkpoint_is_visible_without_fallback():
    adapters = build_adapters(depth=1)
    adapter = adapters["Original Dueling Double DQN"]
    assert adapter.name == "Original Dueling Double DQN"
    assert "loading_status" not in adapter.metadata() or adapter.metadata()["loading_status"] == "loaded"


def test_pause_reset_and_stale_decision_protection():
    state = ArenaState(seed=7)
    legal = state.env.valid_actions()
    decision = Decision(legal[0], legal)
    old_epoch = state.epoch

    state.resume()
    state.reset(seed=8)
    assert state.epoch != old_epoch
    assert state.apply(old_epoch, decision) is False
    assert state.move_count == 0

    state.resume()
    current_epoch = state.epoch
    assert state.apply(current_epoch, decision) is True
    assert state.move_count == 1
    state.pause()
    assert state.apply(current_epoch, decision) is False


def test_completed_game_can_restart():
    state = ArenaState(seed=9)
    state.env.done = True
    state.reset(seed=10)
    assert state.env.done is False
    assert state.move_count == 0
