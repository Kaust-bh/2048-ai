"""Model adapters for the interactive 2048 AI Arena."""

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from agent import DQNAgent
from expectimax_agent import ExpectimaxAgent
from game import ACTIONS, Game2048, strategy_score
from heuristic_agent import HeuristicAgent
from legacy_agent import LegacyDQNAgent, action_mapping_metadata


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


class UnavailableAdapter(ArenaAdapter):
    def __init__(self, name, error):
        self.name = name
        self.error = error

    def decide(self, board, legal_actions):
        raise RuntimeError(f"{self.name} unavailable: {self.error}")

    def metadata(self):
        return {**super().metadata(), "loading_status": "unavailable", "error": self.error}


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
            action, list(legal_actions), values,
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
        return Decision(action, list(legal_actions), values, {
            "search_depth": self.agent.depth,
            **stats,
        }, stats["decision_seconds"])

    def metadata(self):
        return {**super().metadata(), **self.agent.metadata()}


class DQNAdapter(ArenaAdapter):
    def __init__(self, name, checkpoint_path):
        self.name = name
        self.checkpoint_path = str(checkpoint_path)
        self.agent = DQNAgent(seed=42)
        self.checkpoint = self.agent.load(self.checkpoint_path)
        self.agent.epsilon = 0.0
        self.agent.online.eval()

    def decide(self, board, legal_actions):
        state = Game2048(seed=0)
        state.board = np.asarray(board, dtype=np.int64).copy()
        vector = state.state_vector()
        started = time.perf_counter()
        with torch.no_grad():
            tensor = torch.as_tensor(vector, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            raw = self.agent.online(tensor).squeeze(0).detach().cpu().numpy()
        values = {action: float(raw[action]) for action in range(4)}
        masked = {action: values[action] for action in legal_actions}
        action = max(masked, key=masked.get) if masked else None
        metadata = self.checkpoint.get("metadata", {})
        return Decision(action, list(legal_actions), values, {
            "raw_q_values": values,
            "masked_q_values": masked,
            "action_masked": True,
            "checkpoint_path": self.checkpoint_path,
            "architecture": self.checkpoint.get("architecture", metadata.get("model", "unknown")),
            "reward_mode": metadata.get("reward_mode", "unknown"),
            "loading_status": "loaded",
        }, time.perf_counter() - started)

    def metadata(self):
        metadata = self.checkpoint.get("metadata", {})
        return {**super().metadata(), "checkpoint_path": self.checkpoint_path,
                "architecture": self.checkpoint.get("architecture", metadata.get("model", "unknown")),
                "reward_mode": metadata.get("reward_mode", "unknown")}


class LegacyAdapter(ArenaAdapter):
    name = "Strategy V1 DQN (mapping unverified)"

    def __init__(self, checkpoint_path):
        self.checkpoint_path = str(checkpoint_path)
        self.agent = LegacyDQNAgent.load(self.checkpoint_path, device=torch.device("cpu"))

    def decide(self, board, legal_actions):
        state = np.zeros((4, 4), dtype=np.float32)
        values = np.asarray(board, dtype=np.int64)
        mask = values > 0
        state[mask] = np.log2(values[mask]).astype(np.float32) / 16.0
        vector = state.reshape(-1)
        started = time.perf_counter()
        with torch.no_grad():
            tensor = torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0)
            raw = self.agent.network(tensor).squeeze(0).numpy()
        action = self.agent.act(vector, explore=False, valid_actions=legal_actions)
        return Decision(action, list(legal_actions), {i: float(raw[i]) for i in range(4)}, {
            "raw_q_values": {i: float(raw[i]) for i in range(4)},
            "action_masked": True,
            "checkpoint_path": self.checkpoint_path,
            **action_mapping_metadata(),
        }, time.perf_counter() - started)

    def metadata(self):
        return {**super().metadata(), "checkpoint_path": self.checkpoint_path,
                "architecture": "legacy_feedforward_dqn", **action_mapping_metadata()}


def project_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parent / relative_path


def build_adapters(depth=2) -> Dict[str, ArenaAdapter]:
    adapters: Dict[str, ArenaAdapter] = {
        "Greedy Heuristic": HeuristicAdapter(),
        "Expectimax": ExpectimaxAdapter(depth),
    }
    checkpoint_specs = (
        ("Original Dueling Double DQN", "checkpoints_dueling/best_dqn_2048.pt"),
        ("Raw-Score Dueling DQN", "checkpoints_dueling_raw_score/best_dqn_2048.pt"),
    )
    for name, relative in checkpoint_specs:
        path = project_path(relative)
        if path.exists():
            try:
                adapters[name] = DQNAdapter(name, path)
            except Exception as exc:
                adapters[name] = UnavailableAdapter(name, str(exc))
        else:
            adapters[name] = UnavailableAdapter(name, f"missing checkpoint: {path}")
    legacy_path = project_path("backups/strategy_dqn_v1.pt")
    if legacy_path.exists():
        try:
            adapters["Strategy V1 DQN"] = LegacyAdapter(legacy_path)
        except Exception as exc:
            adapters["Strategy V1 DQN"] = UnavailableAdapter("Strategy V1 DQN", str(exc))
    else:
        adapters["Strategy V1 DQN"] = UnavailableAdapter("Strategy V1 DQN", f"missing checkpoint: {legacy_path}")
    return adapters


class ArenaState:
    """Headless game/controller state with stale-decision protection."""

    def __init__(self, seed=None):
        self.seed = seed
        self.env = Game2048(seed=seed)
        self.epoch = 0
        self.paused = True
        self.last_decision: Optional[Decision] = None
        self.move_count = 0

    def reset(self, seed=None):
        self.seed = seed if seed is not None else self.seed
        self.env = Game2048(seed=self.seed)
        self.epoch += 1
        self.last_decision = None
        self.move_count = 0

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def apply(self, epoch, decision):
        if epoch != self.epoch or self.paused or self.env.done:
            return False
        if decision.action not in self.env.valid_actions():
            return False
        self.env.step(decision.action)
        self.move_count += 1
        self.last_decision = decision
        return True
