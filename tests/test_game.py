import numpy as np
import pytest
from game import Game2048, _merge_line, strategy_score


def test_merge_line_once():
    got, points = _merge_line([2, 2, 2, 2])
    assert got.tolist() == [4, 4, 0, 0]
    assert points == 8


def test_merge_does_not_double_merge():
    got, points = _merge_line([2, 2, 4, 0])
    assert got.tolist() == [4, 4, 0, 0]
    assert points == 4


def test_left_move():
    board = np.array([[0, 2, 0, 2], [4, 4, 4, 0], [0, 0, 0, 0], [8, 0, 8, 8]])
    result, gained = Game2048._move_board(board, "left")
    assert result[0].tolist() == [4, 0, 0, 0]
    assert result[1].tolist() == [8, 4, 0, 0]
    assert result[3].tolist() == [16, 8, 0, 0]
    assert gained == 4 + 8 + 16


def test_invalid_move_does_not_spawn():
    env = Game2048(seed=3)
    env.board = np.array([[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]])
    env.score = 0
    before = env.board.copy()
    _, reward, done, info = env.step("left")
    assert np.array_equal(env.board, before)
    assert info["invalid_move"] is True
    assert reward == -0.1


def test_spawn_has_two_tiles_on_reset():
    env = Game2048(seed=9)
    assert np.count_nonzero(env.board) == 2
    assert set(env.board[env.board > 0]).issubset({2, 4})


def test_state_vector_shape_and_values():
    env = Game2048(seed=2)
    vector = env.state_vector()
    assert vector.shape == (16,)
    assert np.all(vector >= 0)
    assert np.all(vector <= 1)


def test_default_reward_mode_preserves_shaped_formula():
    env = Game2048(seed=12)
    env.board = np.array(
        [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    before = env.board.copy()
    _, reward, done, info = env.step("left")
    expected = float(np.log2(info["score_gained"])) - 0.01
    expected += 0.2 * (info["strategy_score"] - strategy_score(before))
    if done:
        expected -= 2.0
    assert env.reward_mode == "shaped"
    assert reward == expected


def test_raw_score_merge_and_non_merge_rewards():
    env = Game2048(seed=13, reward_mode="raw_score")
    env.board = np.array(
        [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    _, merge_reward, _, merge_info = env.step("left")
    assert merge_reward == float(merge_info["score_gained"]) == 4.0

    env.board = np.array(
        [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    _, move_reward, _, move_info = env.step("left")
    assert move_info["score_gained"] == 0
    assert move_reward == 0.0


def test_raw_score_invalid_move_has_zero_reward_and_no_spawn():
    env = Game2048(seed=14, reward_mode="raw_score")
    env.board = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
    )
    before = env.board.copy()
    _, reward, _, info = env.step("left")
    assert reward == 0.0
    assert info["invalid_move"] is True
    assert np.array_equal(env.board, before)


def test_raw_score_terminal_move_has_no_terminal_penalty():
    env = Game2048(seed=15, reward_mode="raw_score")
    env.board = np.array(
        [[2, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]]
    )
    env._spawn_tile = lambda: False
    env.valid_actions = lambda: []
    _, reward, done, info = env.step("left")
    assert done is True
    assert reward == float(info["score_gained"])


def test_invalid_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        Game2048(reward_mode="unknown")
