"""Session and control logic for the browser-based 2048 application.

This module is independent of FastAPI so its state transitions can be tested
without starting an HTTP server. Each session owns one Game2048 instance; AI
decisions are made from a board snapshot and committed only if the session
version and control generation are still current.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

import numpy as np

from web_agents import (
    ArenaAdapter,
    Decision,
    ExpectimaxAdapter,
    HeuristicAdapter,
)
from game import ACTIONS, Game2048


class GameServiceError(Exception):
    """Base error with an HTTP-friendly status code."""

    status_code = 400


class MissingGame(GameServiceError):
    status_code = 404


class InvalidRequest(GameServiceError):
    status_code = 422


class StateConflict(GameServiceError):
    status_code = 409


class AgentFailure(GameServiceError):
    status_code = 503


@dataclass
class Snapshot:
    board: np.ndarray
    score: int
    done: bool
    rng_state: dict
    move_count: int


@dataclass
class GameSession:
    session_id: str
    env: Game2048
    version: int = 0
    move_count: int = 0
    best_score: int = 0
    control: str = "human"
    agent_name: str = "Expectimax"
    depth: int = 2
    thinking: bool = False
    generation: int = 0
    undo: list[Snapshot] = field(default_factory=list)
    last_decision: Optional[dict] = None
    last_error: Optional[str] = None
    last_access: float = field(default_factory=time.monotonic)


AGENT_DESCRIPTIONS = {
    "Greedy Heuristic": "Chooses the legal move with the strongest immediate board heuristic.",
    "Expectimax": "Searches player moves and averages the 2 (90%) / 4 (10%) tile-spawn outcomes.",
}


def build_web_adapters() -> Dict[str, ArenaAdapter]:
    """Create the CPU-only agents supported by the browser application."""
    return {
        "Greedy Heuristic": HeuristicAdapter(),
        "Expectimax": ExpectimaxAdapter(depth=2),
    }


class GameService:
    """Thread-safe in-memory game sessions and optimistic AI move commits."""

    SESSION_TTL_SECONDS = 3 * 60 * 60
    MAX_SESSIONS = 2000

    def __init__(
        self,
        adapters: Optional[Mapping[str, ArenaAdapter]] = None,
        benchmark_path: Optional[Path] = None,
    ):
        self._lock = threading.RLock()
        self._sessions: Dict[str, GameSession] = {}
        self.adapters = dict(adapters if adapters is not None else build_web_adapters())
        self.benchmark_path = benchmark_path or (
            Path(__file__).resolve().parent
            / "results/canonical"
        )

    def _get(self, session_id: str) -> GameSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise MissingGame("game session not found")
        session.last_access = time.monotonic()
        return session

    @staticmethod
    def _snapshot(session: GameSession) -> Snapshot:
        return Snapshot(
            board=session.env.board.copy(),
            score=int(session.env.score),
            done=bool(session.env.done),
            rng_state=deepcopy(session.env.rng.bit_generator.state),
            move_count=session.move_count,
        )

    @staticmethod
    def _restore(session: GameSession, snapshot: Snapshot) -> None:
        session.env.board = snapshot.board.copy()
        session.env.score = snapshot.score
        session.env.done = snapshot.done
        session.env.rng.bit_generator.state = deepcopy(snapshot.rng_state)
        session.move_count = snapshot.move_count

    def _agent_options(self) -> list[dict]:
        options = []
        for name in self.adapters:
            if name not in AGENT_DESCRIPTIONS:
                continue
            options.append({
                "name": name,
                "available": True,
                "description": AGENT_DESCRIPTIONS[name],
                "reason": None,
            })
        return options

    def _serialize(self, session: GameSession) -> dict:
        board = session.env.board
        largest = int(board.max())
        if session.env.done:
            status = "finished"
        elif session.last_error:
            status = "error"
        elif session.thinking:
            status = "thinking"
        elif session.control == "ai":
            status = "playing"
        elif session.version == 0 or session.move_count == 0:
            status = "ready"
        else:
            status = "stopped" if session.last_decision else "human"
        return {
            "id": session.session_id,
            "version": session.version,
            "board": board.astype(int).tolist(),
            "score": int(session.env.score),
            "best_score": int(session.best_score),
            "move_count": int(session.move_count),
            "max_tile": largest,
            "won": largest >= 2048,
            "game_over": bool(session.env.done),
            "control": session.control,
            "status": status,
            "agent": session.agent_name,
            "depth": session.depth,
            "can_undo": bool(session.undo),
            "last_decision": session.last_decision,
            "error": session.last_error,
            "valid_actions": [ACTIONS[i] for i in session.env.valid_actions()],
            "agents": self._agent_options(),
        }

    def new_game(self, seed: Optional[int] = None) -> dict:
        with self._lock:
            now = time.monotonic()
            expired = [
                key for key, value in self._sessions.items()
                if now - value.last_access > self.SESSION_TTL_SECONDS
            ]
            for key in expired:
                del self._sessions[key]
            if len(self._sessions) >= self.MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].last_access)
                del self._sessions[oldest]

            session_id = uuid4().hex
            session = GameSession(
                session_id=session_id,
                env=Game2048(seed=seed),
            )
            self._sessions[session_id] = session
            return self._serialize(session)

    def get_game(self, session_id: str) -> dict:
        with self._lock:
            return self._serialize(self._get(session_id))

    def move(self, session_id: str, action: str, expected_version: int) -> dict:
        with self._lock:
            session = self._get(session_id)
            if expected_version != session.version:
                raise StateConflict("board changed; refresh before moving")
            if session.control != "human":
                raise StateConflict("take control before making a human move")
            if session.env.done:
                raise StateConflict("game is over")
            if action not in ACTIONS:
                raise InvalidRequest("action must be up, right, down, or left")
            action_id = ACTIONS.index(action)
            if action_id not in session.env.valid_actions():
                raise InvalidRequest("that move does not change the board")

            session.undo.append(self._snapshot(session))
            _, _, _, _ = session.env.step(action_id)
            session.move_count += 1
            session.best_score = max(session.best_score, int(session.env.score))
            session.version += 1
            session.last_decision = {"action": action, "source": "human"}
            session.last_error = None
            return self._serialize(session)

    def set_control(self, session_id: str, control: str) -> dict:
        if control not in ("human", "ai"):
            raise InvalidRequest("control must be 'human' or 'ai'")
        with self._lock:
            session = self._get(session_id)
            if control == "ai" and session.env.done:
                raise StateConflict("start a new game before asking AI to play")
            if control == "ai":
                adapter = self.adapters.get(session.agent_name)
                if adapter is None:
                    raise InvalidRequest(f"{session.agent_name} is unavailable")
            session.generation += 1
            session.version += 1
            session.control = control
            session.thinking = False
            session.last_error = None
            return self._serialize(session)

    def configure_agent(self, session_id: str, name: str, depth: int) -> dict:
        with self._lock:
            session = self._get(session_id)
            if name not in self.adapters:
                raise InvalidRequest("unknown agent")
            if depth not in (1, 2, 3):
                raise InvalidRequest("Expectimax depth must be between 1 and 3")
            session.agent_name = name
            session.depth = depth
            session.generation += 1
            session.version += 1
            session.control = "human"
            session.thinking = False
            session.last_error = None
            return self._serialize(session)

    def _decision_adapter(self, name: str, depth: int) -> ArenaAdapter:
        adapter = self.adapters.get(name)
        if adapter is None:
            raise AgentFailure(f"{name} is unavailable")
        if name == "Expectimax" and isinstance(adapter, ExpectimaxAdapter):
            return ExpectimaxAdapter(depth)
        return adapter

    def ai_move(self, session_id: str, expected_version: int) -> dict:
        with self._lock:
            session = self._get(session_id)
            if expected_version != session.version:
                raise StateConflict("board changed before the AI request")
            if session.control != "ai":
                raise StateConflict("AI control is not active")
            if session.thinking:
                raise StateConflict("an AI decision is already in progress")
            if session.env.done:
                raise StateConflict("game is over")
            adapter = self._decision_adapter(session.agent_name, session.depth)
            version = session.version
            generation = session.generation
            agent_name = session.agent_name
            depth = session.depth
            board = session.env.board.copy()
            legal = session.env.valid_actions()
            session.thinking = True

        try:
            decision = adapter.decide(board, legal)
        except Exception as exc:
            with self._lock:
                current = self._sessions.get(session_id)
                if current is not None and current.version == version and current.generation == generation:
                    current.thinking = False
                    current.control = "human"
                    current.generation += 1
                    current.version += 1
                    current.last_error = f"{type(exc).__name__}: {exc}"
            raise AgentFailure(f"{agent_name} failed: {type(exc).__name__}: {exc}") from exc

        with self._lock:
            session = self._get(session_id)
            if (
                session.version != version
                or session.generation != generation
                or session.control != "ai"
                or session.agent_name != agent_name
                or not np.array_equal(session.env.board, board)
            ):
                raise StateConflict("AI decision became stale; no move was applied")
            if decision.action not in legal:
                session.thinking = False
                session.control = "human"
                session.generation += 1
                session.version += 1
                session.last_error = "Agent returned an illegal action; board was not changed."
                raise AgentFailure(session.last_error)

            session.undo.append(self._snapshot(session))
            session.env.step(int(decision.action))
            session.move_count += 1
            session.best_score = max(session.best_score, int(session.env.score))
            session.version += 1
            session.thinking = False
            session.last_error = None
            session.last_decision = {
                "action": decision.action_name,
                "source": "ai",
                "agent": agent_name,
                "latency_ms": round(decision.latency_seconds * 1000, 2),
                "values": {ACTIONS[int(k)]: float(v) for k, v in decision.values.items()},
                "diagnostics": decision.diagnostics,
            }
            return self._serialize(session)

    def undo_move(self, session_id: str) -> dict:
        with self._lock:
            session = self._get(session_id)
            if not session.undo:
                raise StateConflict("there is no move to undo")
            session.generation += 1
            session.control = "human"
            session.thinking = False
            snapshot = session.undo.pop()
            self._restore(session, snapshot)
            session.version += 1
            session.last_error = None
            session.last_decision = None
            return self._serialize(session)

    def reset(self, session_id: str, seed: Optional[int] = None) -> dict:
        with self._lock:
            session = self._get(session_id)
            session.generation += 1
            session.env = Game2048(seed=seed)
            session.move_count = 0
            session.undo.clear()
            session.control = "human"
            session.thinking = False
            session.last_decision = None
            session.last_error = None
            session.version += 1
            return self._serialize(session)

    def benchmarks(self) -> dict:
        summaries = {
            "Expectimax depth 2": ("expectimax_depth_2_seeds", "expectimax_strategy_score", "expectimax"),
            "Greedy Heuristic": ("heuristic_strategy_v1_seeds", "heuristic_strategy_score", "hand_coded_strategy_score"),
        }
        rows = []
        for name, (folder, agent_id, architecture) in summaries.items():
            path = self.benchmark_path / folder / "summary.json"
            try:
                document = json.loads(path.read_text())
                summary = document["summary"]
                metadata = document["metadata"]
                evaluation = metadata["evaluation"]
                if (
                    metadata["agent_id"] != agent_id
                    or metadata["architecture"] != architecture
                    or evaluation["games"] != 100
                    or evaluation["max_moves"] != 5000
                    or evaluation["seeds"] != list(range(1000, 1100))
                    or evaluation["exploration"] != "disabled"
                    or summary["games"] != 100
                ):
                    continue
                max_tile = max(int(value) for value in summary["max_tile_distribution"])
                rows.append({
                    "agent": name,
                    "games": summary["games"],
                    "mean_score": summary["mean_score"],
                    "mean_moves": summary["mean_moves"],
                    "max_tile_observed": max_tile,
                    "reached_2048_pct": summary["reached_2048_pct"],
                })
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return {
            "conditions": "Historical canonical runs: 100 games, seeds 1000–1099, 5,000 move cap, exploration disabled. Live games are separate.",
            "rng_note": "Seeds reproduce initial boards; policies can consume different random streams after their move histories diverge.",
            "rows": rows,
        }
