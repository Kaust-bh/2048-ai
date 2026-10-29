"""Expectimax policy for 2048 using the existing board heuristic."""

from typing import Dict, Optional, Sequence, Tuple
import time

import numpy as np

from game import Game2048, strategy_score


SPAWN_PROBABILITIES = ((2, 0.9), (4, 0.1))
ACTIONS = (0, 1, 2, 3)


class ExpectimaxAgent:
    """Search player moves and random spawns with a heuristic leaf value.

    ``depth`` counts player moves. A depth-one search evaluates each legal
    move after averaging every possible tile spawn. Deeper searches repeat
    player-max and chance-expectation layers. Search never calls ``step`` or
    the environment RNG.
    """

    def __init__(self, depth: int = 2, use_cache: bool = True):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = int(depth)
        self.use_cache = use_cache
        self._cache: Dict[Tuple[str, int, bytes], float] = {}
        self._reset_stats()

    def _reset_stats(self):
        self.last_search_nodes = 0
        self.last_cache_hits = 0
        self.last_player_nodes = 0
        self.last_chance_nodes = 0
        self.last_leaf_evaluations = 0
        self.last_max_empty_branching = 0
        self.last_decision_seconds = 0.0

    def search_stats(self):
        """Return a copy of statistics for the most recent decision."""
        return {
            "nodes": self.last_search_nodes,
            "cache_hits": self.last_cache_hits,
            "player_nodes": self.last_player_nodes,
            "chance_nodes": self.last_chance_nodes,
            "leaf_evaluations": self.last_leaf_evaluations,
            "max_empty_branching": self.last_max_empty_branching,
            "decision_seconds": self.last_decision_seconds,
        }

    @staticmethod
    def _legal_actions(board: np.ndarray):
        legal = []
        for action in ACTIONS:
            moved, _ = Game2048._move_board(board, action)
            if not np.array_equal(moved, board):
                legal.append(action)
        return legal

    @staticmethod
    def _cache_key(node_type: str, board: np.ndarray, depth: int):
        return node_type, depth, np.asarray(board, dtype=np.int64).tobytes()

    def _leaf_value(self, board: np.ndarray) -> float:
        return float(strategy_score(board))

    def _chance_value(self, board: np.ndarray, depth: int) -> float:
        key = self._cache_key("C", board, depth)
        if self.use_cache and key in self._cache:
            self.last_cache_hits += 1
            return self._cache[key]

        self.last_search_nodes += 1
        self.last_chance_nodes += 1
        empty = np.argwhere(board == 0)
        self.last_max_empty_branching = max(
            self.last_max_empty_branching, len(empty)
        )
        if len(empty) == 0:
            value = self._player_value(board, depth - 1)
        else:
            value = 0.0
            cell_probability = 1.0 / len(empty)
            for row, column in empty:
                for tile, tile_probability in SPAWN_PROBABILITIES:
                    spawned = np.asarray(board, dtype=np.int64).copy()
                    spawned[row, column] = tile
                    value += (
                        cell_probability
                        * tile_probability
                        * self._player_value(spawned, depth - 1)
                    )

        if self.use_cache:
            self._cache[key] = value
        return value

    def _player_value(self, board: np.ndarray, depth: int) -> float:
        key = self._cache_key("P", board, depth)
        if self.use_cache and key in self._cache:
            self.last_cache_hits += 1
            return self._cache[key]

        self.last_search_nodes += 1
        self.last_player_nodes += 1
        legal_actions = self._legal_actions(board)
        if depth <= 0 or not legal_actions:
            self.last_leaf_evaluations += 1
            value = self._leaf_value(board)
        else:
            value = max(
                self._chance_value(
                    Game2048._move_board(board, action)[0],
                    depth,
                )
                for action in legal_actions
            )

        if self.use_cache:
            self._cache[key] = value
        return value

    def action_values(
        self,
        board: np.ndarray,
        valid_actions: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        """Return expected values for legal actions in deterministic order."""
        source = np.asarray(board, dtype=np.int64)
        self._cache.clear()
        self._reset_stats()
        legal_actions = self._legal_actions(source)
        if valid_actions is not None:
            supplied = set(valid_actions)
            legal_actions = [action for action in legal_actions if action in supplied]
        if not legal_actions:
            return {}

        started = time.perf_counter()
        values = {
            action: self._chance_value(
                Game2048._move_board(source, action)[0],
                self.depth,
            )
            for action in legal_actions
        }
        self.last_decision_seconds = time.perf_counter() - started
        return values

    def select_action(
        self,
        board: np.ndarray,
        valid_actions: Optional[Sequence[int]] = None,
    ) -> Optional[int]:
        """Return the best legal action, or None for a terminal board."""
        values = self.action_values(board, valid_actions)
        if not values:
            return None
        return max(values, key=lambda action: values[action])

    def __call__(self, state, valid_actions, board):
        del state
        return self.select_action(board, valid_actions)

    def metadata(self):
        return {
            "agent_id": "expectimax_strategy_score",
            "architecture": "expectimax",
            "search_depth": self.depth,
            "depth_convention": "one unit equals one player move plus its chance spawn",
            "spawn_probabilities": {str(tile): probability for tile, probability in SPAWN_PROBABILITIES},
            "cache_enabled": self.use_cache,
        }
