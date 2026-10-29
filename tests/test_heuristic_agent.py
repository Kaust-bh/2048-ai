import json

import numpy as np

from canonical_evaluation import EvaluationConfig, evaluate_policy, write_results
from game import Game2048, strategy_score
from heuristic_agent import HeuristicAgent


def test_policy_only_selects_valid_actions():
    env = Game2048(seed=17)
    agent = HeuristicAgent()

    action = agent.select_action(env.board, env.valid_actions())

    assert action in env.valid_actions()


def test_action_selection_does_not_mutate_board():
    env = Game2048(seed=18)
    agent = HeuristicAgent()
    before = env.board.copy()

    agent.select_action(env.board, env.valid_actions())

    assert np.array_equal(env.board, before)


def test_repeated_calls_are_deterministic():
    env = Game2048(seed=19)
    agent = HeuristicAgent()
    valid_actions = env.valid_actions()

    actions = [
        agent.select_action(env.board, valid_actions)
        for _ in range(5)
    ]

    assert actions == [actions[0]] * 5


def test_action_has_highest_immediate_strategy_score():
    board = np.array(
        [
            [2, 0, 0, 0],
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    agent = HeuristicAgent()
    valid_actions = agent._valid_actions(board)
    scores = {
        action: strategy_score(Game2048._move_board(board, action)[0])
        for action in valid_actions
    }

    action = agent.select_action(board, valid_actions)

    assert scores[action] == max(scores.values())


def test_terminal_board_returns_no_action():
    board = np.array(
        [
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
        ],
        dtype=np.int64,
    )
    agent = HeuristicAgent()

    assert agent.select_action(board, []) is None
    assert agent.select_action(board) is None


def test_canonical_evaluator_records_heuristic_identity(tmp_path):
    agent = HeuristicAgent()
    rows, metadata = evaluate_policy(
        agent,
        EvaluationConfig(seeds=(41, 42), max_moves=10),
        agent_id="heuristic_strategy_score",
        architecture="hand_coded_strategy_score",
    )
    csv_path, json_path = write_results(
        str(tmp_path), rows, metadata
    )

    with open(json_path) as handle:
        document = json.load(handle)

    assert all(
        row["agent_id"] == "heuristic_strategy_score"
        for row in rows
    )
    assert document["metadata"]["agent_id"] == "heuristic_strategy_score"
    assert document["metadata"]["architecture"] == "hand_coded_strategy_score"
    assert csv_path.endswith("games.csv")