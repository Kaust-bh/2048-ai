"""CPU-only adapters for the browser demo's supported agents."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from expectimax_agent import ExpectimaxAgent
from game import ACTIONS, Game2048, strategy_score
from heuristic_agent import HeuristicAgent


ACTION_NAMES = tuple(ACTIONS)


@dataclass
class Decision:
    action: Optional[int]
    legal_actions: list
    values: Dict[int, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0

    @property
    def action_name(self):
        return ACTION_NAMES[self.action] if self.action is not None else "none"


class ArenaAdapter:
    name = "Model"

    def decide(self, board: np.ndarray, legal_actions: Sequence[int]) -> Decision:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {"agent_id": self.name, "action_mapping": dict(enumerate(ACTION_NAMES))}


class HeuristicAdapter(ArenaAdapter):
    name = "Greedy Heuristic"

    def __init__(self):
        self.agent = HeuristicAgent()

    def decide(self, board, legal_actions):
        started = time.perf_counter()
        values = {}
        for action in legal_actions:
            moved, _ = Game2048._move_board(board, action)
            values[action] = strategy_score(moved)
        action = self.agent.select_action(board, legal_actions)
        return Decision(
            action,
            list(legal_actions),
            values,
            {"strategy_score": strategy_score(board), "selection": "highest immediate strategy_score"},
            time.perf_counter() - started,
        )


class ExpectimaxAdapter(ArenaAdapter):
    name = "Expectimax"

    def __init__(self, depth=2):
        self.agent = ExpectimaxAgent(depth=depth)

    def decide(self, board, legal_actions):
        values = self.agent.action_values(board, legal_actions)
        action = max(values, key=values.get) if values else None
        stats = self.agent.search_stats()
        return Decision(
            action,
            list(legal_actions),
            {int(key): float(value) for key, value in values.items()},
            {"search_depth": self.agent.depth, **stats},
            stats["decision_seconds"],
        )

    def metadata(self):
        return {**super().metadata(), **self.agent.metadata()}
