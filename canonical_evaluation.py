"""Shared, reproducible evaluation for 2048 policies.

A seed fixes the initial board and the environment RNG stream. Because
Game2048 draws random numbers when spawning tiles, policies that take
different numbers of moves consume different portions of that stream.
Therefore identical seeds guarantee identical initial boards, but do not
guarantee identical later tile sequences after action histories diverge.
"""

from collections import Counter
from dataclasses import dataclass
import csv
import inspect
import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from game import Game2048


STATE_REPRESENTATION_VERSION = "log2_tile_v1"
CSV_FIELDS = (
    "game",
    "seed",
    "score",
    "max_tile",
    "moves",
    "reached_2048",
    "agent_id",
    "checkpoint_path",
)

Policy = Callable[[np.ndarray, Sequence[int]], int]


def _call_policy(policy, state, valid_actions, board):
    """Call legacy two-argument or board-aware policy callbacks."""
    try:
        inspect.signature(policy).bind(state, valid_actions, board)
    except (TypeError, ValueError):
        return policy(state, valid_actions)
    return policy(state, valid_actions, board)


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: Tuple[int, ...]
    max_moves: int = 5000

    @property
    def games(self) -> int:
        return len(self.seeds)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "games": self.games,
            "max_moves": self.max_moves,
            "exploration": "disabled",
            "environment": "Game2048",
            "state_representation_version": STATE_REPRESENTATION_VERSION,
            "rng_note": (
                "Seeds reproduce the initial board. Later tile draws can "
                "differ when policies take different numbers of moves."
            ),
        }


def make_seed_list(games: int = 100, seed_start: int = 1000) -> List[int]:
    if games < 1:
        raise ValueError("games must be positive")
    return list(range(seed_start, seed_start + games))


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty result set")

    scores = np.asarray([int(row["score"]) for row in rows])
    moves = np.asarray([int(row["moves"]) for row in rows])
    max_tiles = [int(row["max_tile"]) for row in rows]
    reached = sum(bool(row["reached_2048"]) for row in rows)

    return {
        "games": len(rows),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "score_stddev": float(np.std(scores)),
        "mean_moves": float(np.mean(moves)),
        "max_tile_distribution": {
            str(tile): count
            for tile, count in sorted(Counter(max_tiles).items())
        },
        "reached_512_pct": 100.0 * sum(
            tile >= 512 for tile in max_tiles
        ) / len(rows),
        "reached_1024_pct": 100.0 * sum(
            tile >= 1024 for tile in max_tiles
        ) / len(rows),
        "reached_2048_pct": 100.0 * reached / len(rows),
    }


def evaluate_policy(
    policy: Policy,
    config: EvaluationConfig,
    agent_id: str,
    checkpoint_path: Optional[str] = None,
    architecture: str = "unknown",
    on_game_complete=None,
    on_progress=None,
    progress_interval: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate a policy using the same Game2048 mechanics for every seed."""
    if config.max_moves < 1:
        raise ValueError("max_moves must be positive")
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")

    rows = []
    for game_number, seed in enumerate(config.seeds, start=1):
        env = Game2048(seed=seed)
        state = env.state_vector()
        moves = 0

        while not env.done and moves < config.max_moves:
            valid_actions = env.valid_actions()
            if not valid_actions:
                break

            action = _call_policy(
                policy,
                state,
                valid_actions,
                env.board.copy(),
            )
            if action is None:
                raise ValueError(
                    "policy returned no action while valid actions exist"
                )
            action = int(action)
            if action not in valid_actions:
                raise ValueError(
                    f"policy returned invalid action {action}; "
                    f"valid actions are {valid_actions}"
                )

            _, _, _, _ = env.step(action)
            state = env.state_vector()
            moves += 1

        max_tile = int(env.board.max())
        rows.append({
            "game": game_number,
            "seed": int(seed),
            "score": int(env.score),
            "max_tile": max_tile,
            "moves": moves,
            "reached_2048": max_tile >= 2048,
            "agent_id": agent_id,
            "checkpoint_path": checkpoint_path or "",
        })
        if on_game_complete is not None:
            on_game_complete(rows[-1])
        completed = game_number
        if (
            on_progress is not None
            and (
                completed % progress_interval == 0
                or completed == config.games
            )
        ):
            on_progress(completed, config.games)

    metadata = {
        "agent_id": agent_id,
        "checkpoint_path": checkpoint_path,
        "architecture": architecture,
        "git_revision": _git_revision(),
        "state_representation_version": STATE_REPRESENTATION_VERSION,
        "evaluation": config.as_dict(),
    }
    return rows, metadata


def write_results(
    output_dir: str,
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Tuple[str, str]:
    """Write per-game CSV and metadata/summary JSON, without model files."""
    if not rows:
        raise ValueError("cannot write an empty result set")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "games.csv")
    json_path = os.path.join(output_dir, "summary.json")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    document = {
        "metadata": metadata,
        "summary": summary_from_csv(csv_path),
    }
    with open(json_path, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return csv_path, json_path


def read_results(csv_path: str) -> List[Dict[str, Any]]:
    """Read canonical CSV rows with stable Python types for validation."""
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["game"] = int(row["game"])
        row["seed"] = int(row["seed"])
        row["score"] = int(row["score"])
        row["max_tile"] = int(row["max_tile"])
        row["moves"] = int(row["moves"])
        row["reached_2048"] = row["reached_2048"].lower() == "true"
    return rows


def summary_from_csv(csv_path: str) -> Dict[str, Any]:
    return _summary(read_results(csv_path))
