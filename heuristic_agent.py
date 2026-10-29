"""Hand-coded one-step heuristic policy for 2048."""

from typing import Optional, Sequence

import numpy as np

from game import Game2048, strategy_score


class HeuristicAgent:
    """Choose the legal move with the best immediate strategy score.

    This is deliberately greedy: it scores the board after movement, before
    the environment spawns the next random tile. It is a baseline, not a
    long-term value estimator.
    """

    def select_action(
        self,
        board: np.ndarray,
        valid_actions: Optional[Sequence[int]] = None,
    ) -> Optional[int]:
        """Return the best legal action, or None when no action is legal."""
        source = np.asarray(board, dtype=np.int64)
        legal_actions = self._valid_actions(source)
        if valid_actions is None:
            actions = legal_actions
        else:
            legal_set = set(legal_actions)
            actions = [
                int(action)
                for action in valid_actions
                if action in legal_set
            ]

        if not actions:
            return None

        best_action = None
        best_score = float("-inf")
        for action in actions:
            simulated, _ = Game2048._move_board(source, action)
            score = strategy_score(simulated)

            # Strict comparison preserves ACTIONS order for ties.
            if best_action is None or score > best_score:
                best_action = int(action)
                best_score = score

        return best_action

    @staticmethod
    def _valid_actions(board: np.ndarray):
        actions = []
        for action in range(4):
            simulated, _ = Game2048._move_board(board, action)
            if not np.array_equal(simulated, board):
                actions.append(action)
        return actions

    def __call__(self, state, valid_actions, board):
        """Adapt the policy to the canonical evaluator's board-aware form."""
        del state
        return self.select_action(board, valid_actions)
