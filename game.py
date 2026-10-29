
"""2048 game environment with strategy-based reward shaping."""

import numpy as np

ACTIONS = ("up", "right", "down", "left")
ACTION_TO_DELTA = {name: i for i, name in enumerate(ACTIONS)}

# Highest weight at bottom-left, descending in a snake pattern.
_SNAKE_WEIGHTS = (
    (4,  3,  2,  1),
    (8,  7,  6,  5),
    (9, 10, 11, 12),
    (16, 15, 14, 13),
)

_EMPTY_WEIGHT = 0.30
_SNAKE_WEIGHT = 0.35
_MONOTONICITY_WEIGHT = 0.20
_CORNER_WEIGHT = 0.15


def _merge_line(line):
    """Slide nonzero values left and merge equal adjacent values once."""
    values = [int(x) for x in line if x != 0]
    out, gained, i = [], 0, 0

    while i < len(values):
        if i + 1 < len(values) and values[i] == values[i + 1]:
            merged = values[i] * 2
            out.append(merged)
            gained += merged
            i += 2
        else:
            out.append(values[i])
            i += 1

    out.extend([0] * (len(line) - len(out)))
    return np.asarray(out, dtype=np.int64), gained


def strategy_score(board):
    """
    Fast, bounded heuristic in [0, 1].

    Rewards empty cells, snake ordering, monotonicity,
    and the largest tile in the bottom-left corner.
    """
    logs = [0] * 16
    empty_count = 0
    largest = 0
    max_log = 0

    for r in range(4):
        for c in range(4):
            tile = int(board[r, c])
            idx = r * 4 + c

            if tile == 0:
                empty_count += 1
                continue

            value = tile.bit_length() - 1
            logs[idx] = value

            if tile > largest:
                largest = tile
                max_log = value

    if largest == 0:
        return 1.0

    # Reward larger tiles in higher-weight cells.
    weighted_sum = 0
    for r in range(4):
        for c in range(4):
            weighted_sum += (
                logs[r * 4 + c] * _SNAKE_WEIGHTS[r][c]
            )

    snake_score = min(1.0, weighted_sum / (max_log * 136.0))

    # Penalize changes in direction within rows and columns.
    reversal_penalty = 0

    for r in range(4):
        start = r * 4
        increasing = 0
        decreasing = 0

        for c in range(3):
            diff = logs[start + c + 1] - logs[start + c]

            if diff > 0:
                increasing += diff
            else:
                decreasing -= diff

        reversal_penalty += min(increasing, decreasing)

    for c in range(4):
        increasing = 0
        decreasing = 0

        for r in range(3):
            diff = logs[(r + 1) * 4 + c] - logs[r * 4 + c]

            if diff > 0:
                increasing += diff
            else:
                decreasing -= diff

        reversal_penalty += min(increasing, decreasing)

    monotonicity = 1.0 - min(
        1.0, reversal_penalty / (24.0 * max_log)
    )

    # Reward the largest tile in the chosen corner.
    corner_score = 1.0 if int(board[3, 0]) == largest else 0.0
    empty_score = empty_count / 16.0

    score = (
        _EMPTY_WEIGHT * empty_score
        + _SNAKE_WEIGHT * snake_score
        + _MONOTONICITY_WEIGHT * monotonicity
        + _CORNER_WEIGHT * corner_score
    )

    return max(0.0, min(1.0, score))


class Game2048:
    def __init__(self, seed=None, reward_mode="shaped"):
        if reward_mode not in ("shaped", "raw_score"):
            raise ValueError(
                "reward_mode must be 'shaped' or 'raw_score'"
            )
        self.rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy()

    def _spawn_tile(self):
        """Spawn a 2 or 4 in a random empty cell."""
        empty = np.argwhere(self.board == 0)

        if len(empty) == 0:
            return False

        r, c = empty[self.rng.integers(len(empty))]
        self.board[r, c] = 4 if self.rng.random() < 0.1 else 2
        return True

    @staticmethod
    def _move_board(board, action):
        """Return (new_board, score_gained) for an action."""
        if isinstance(action, str):
            try:
                action = ACTION_TO_DELTA[action.lower()]
            except KeyError as exc:
                raise ValueError(
                    "action must be 0..3 or up, right, down, left"
                ) from exc

        if action not in range(4):
            raise ValueError(
                "action must be 0..3 or up, right, down, left"
            )

        source = np.asarray(board, dtype=np.int64)

        # Action mapping: up=0, right=1, down=2, left=3.
        rotation = (action + 1) % 4
        rotated = np.rot90(source, k=rotation)

        moved = np.zeros_like(rotated)
        gained = 0

        for r in range(4):
            merged, points = _merge_line(rotated[r])
            moved[r] = merged
            gained += points

        result = np.rot90(moved, k=-rotation).copy()
        return result, gained

    def valid_actions(self):
        """Return actions that actually change the board."""
        valid = []

        for action in range(4):
            new_board, _ = self._move_board(self.board, action)

            if not np.array_equal(new_board, self.board):
                valid.append(action)

        return valid

    def step(self, action):
        """Apply a move and return observation, reward, done, info."""
        if self.done:
            raise RuntimeError(
                "Game is over. Call reset() before stepping again."
            )

        new_board, gained = self._move_board(self.board, action)
        changed = not np.array_equal(new_board, self.board)

        if not changed:
            reward = 0.0 if self.reward_mode == "raw_score" else -0.1
            return self.board.copy(), reward, self.done, {
                "score_gained": 0,
                "invalid_move": True,
                "score": self.score,
            }

        old_strategy = strategy_score(self.board)

        self.board = new_board
        self.score += gained

        # Spawn before checking whether the game has ended.
        self._spawn_tile()
        self.done = len(self.valid_actions()) == 0

        new_strategy = strategy_score(self.board)

        if self.reward_mode == "raw_score":
            reward = float(gained)
        else:
            reward = float(np.log2(gained)) if gained > 0 else 0.0
            reward -= 0.01
            reward += 0.2 * (new_strategy - old_strategy)

            if self.done:
                reward -= 2.0

        return self.board.copy(), reward, self.done, {
            "score_gained": gained,
            "invalid_move": False,
            "score": self.score,
            "strategy_score": new_strategy,
        }

    def state_vector(self):
        """Encode tiles as log2(tile)/16; empty cells remain zero."""
        values = np.zeros((4, 4), dtype=np.float32)
        mask = self.board > 0
        values[mask] = (
            np.log2(self.board[mask]).astype(np.float32) / 16.0
        )
        return values.reshape(-1)

    def render_text(self):
        return "\n".join(
            " ".join(f"{int(v):5d}" if v else "    ." for v in row)
            for row in self.board
        )
