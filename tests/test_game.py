import numpy as np
from game import Game2048, _merge_line


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
