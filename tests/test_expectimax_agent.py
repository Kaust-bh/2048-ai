import copy
import json

import numpy as np
import pytest

from canonical_evaluation import EvaluationConfig, evaluate_policy, write_results
from bench_expectimax import run as run_benchmark
from evaluate_expectimax import evaluate
from expectimax_agent import ExpectimaxAgent
from game import Game2048, strategy_score


def expected_spawn_value(board, depth=1):
    empty = np.argwhere(board == 0)
    if not len(empty):
        return strategy_score(board)
    value = 0.0
    for row, column in empty:
        for tile, probability in ((2, 0.9), (4, 0.1)):
            spawned = board.copy()
            spawned[row, column] = tile
            value += probability / len(empty) * strategy_score(spawned)
    return value


def test_depth_one_matches_reference_chance_calculation():
    board = np.array(
        [[2, 2, 4, 8], [16, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=1)
    values = agent.action_values(board)

    reference = {}
    for action in values:
        moved, _ = Game2048._move_board(board, action)
        reference[action] = expected_spawn_value(moved)

    assert values == reference


def test_player_node_selects_maximum_expected_action():
    board = np.array(
        [[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=1)
    values = agent.action_values(board)

    assert agent.select_action(board) == max(values, key=values.get)


def test_spawn_probabilities_and_all_empty_cells_are_modeled():
    board = np.zeros((4, 4), dtype=np.int64)
    agent = ExpectimaxAgent(depth=1)
    moved = board.copy()
    value = agent._chance_value(moved, 1)

    expected = expected_spawn_value(moved)

    assert value == expected
    assert abs(sum(probability for _, probability in ((2, 0.9), (4, 0.1))) - 1.0) < 1e-12


def test_full_board_chance_node_has_no_spawn_branch():
    board = np.array(
        [[2, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=2)

    assert agent._chance_value(board, 2) == agent._player_value(board, 1)


def test_terminal_board_and_invalid_actions():
    board = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=2)

    assert agent.select_action(board) is None
    assert agent.select_action(board, []) is None


def test_legal_action_filtering_and_determinism():
    board = np.array(
        [[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=1)
    valid = agent._legal_actions(board)
    action = agent.select_action(board, valid)

    assert action in valid
    assert action == agent.select_action(board, valid)


def test_search_does_not_mutate_board_or_rng():
    env = Game2048(seed=77)
    before_board = env.board.copy()
    before_rng = copy.deepcopy(env.rng.bit_generator.state)

    ExpectimaxAgent(depth=2).select_action(env.board, env.valid_actions())

    assert np.array_equal(env.board, before_board)
    assert env.rng.bit_generator.state == before_rng


def test_default_depth_and_cache_are_available():
    agent = ExpectimaxAgent()
    assert agent.depth == 2
    assert agent.use_cache is True
    assert agent.select_action(np.array([[2] + [0] * 3] + [[0] * 4] * 3)) in range(4)
    assert agent.last_search_nodes > 0


def test_search_statistics_reset_and_accounting():
    board = np.array(
        [[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=1)

    agent.select_action(board)
    stats = agent.search_stats()

    assert stats["nodes"] == stats["player_nodes"] + stats["chance_nodes"]
    assert stats["leaf_evaluations"] <= stats["player_nodes"]
    assert stats["max_empty_branching"] > 0
    assert stats["decision_seconds"] >= 0.0

    terminal = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]],
        dtype=np.int64,
    )
    assert agent.select_action(terminal) is None
    reset_stats = agent.search_stats()
    assert reset_stats["nodes"] == 0
    assert reset_stats["cache_hits"] == 0


def test_cache_hits_are_counted_for_repeated_state():
    board = np.array(
        [[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    )
    agent = ExpectimaxAgent(depth=1)
    moved, _ = Game2048._move_board(board, 0)
    agent._chance_value(moved, 1)
    before = agent.search_stats()["cache_hits"]
    agent._chance_value(moved, 1)

    assert agent.search_stats()["cache_hits"] == before + 1


def test_canonical_expectimax_integration(tmp_path):
    agent = ExpectimaxAgent(depth=1)
    rows, metadata = evaluate_policy(
        agent,
        EvaluationConfig(seeds=(81, 82), max_moves=8),
        agent_id="expectimax_strategy_score",
        architecture="expectimax",
    )
    metadata.update(agent.metadata())
    _, json_path = write_results(str(tmp_path), rows, metadata)

    with open(json_path) as handle:
        document = json.load(handle)

    assert len(rows) == 2
    assert document["metadata"]["search_depth"] == 1
    assert document["metadata"]["depth_convention"].startswith("one unit")
    assert document["metadata"]["spawn_probabilities"] == {"2": 0.9, "4": 0.1}


def test_incremental_cli_outputs_and_existing_directory_refusal(tmp_path):
    output_dir = tmp_path / "expectimax"
    evaluate(
        depth=1,
        seeds=(91, 92),
        max_moves=2,
        output_dir=str(output_dir),
    )

    with (output_dir / "games.csv").open() as handle:
        assert len(handle.readlines()) == 3
    with (output_dir / "summary.json").open() as handle:
        document = json.load(handle)
    assert document["summary"]["games"] == 2

    with pytest.raises(FileExistsError):
        evaluate(
            depth=1,
            seeds=(91,),
            max_moves=1,
            output_dir=str(output_dir),
        )


def test_benchmark_reports_fixed_board_statistics(capsys):
    run_benchmark(depths=(1,), decisions=1)
    output = capsys.readouterr().out
    assert "board=sparse depth=1" in output
    assert "cache_hits=" in output
    assert "max_seconds=" in output
