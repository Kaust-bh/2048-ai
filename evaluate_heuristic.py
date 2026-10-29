"""Evaluate the hand-coded heuristic with canonical evaluation settings."""

import argparse
import os

from canonical_evaluation import (
    EvaluationConfig,
    evaluate_policy,
    make_seed_list,
    write_results,
)
from heuristic_agent import HeuristicAgent


def evaluate(
    games=100,
    seed=1000,
    max_moves=5000,
    seeds=None,
    output_dir="results/canonical/heuristic",
):
    evaluation_seeds = tuple(
        int(value) for value in (
            seeds if seeds is not None else make_seed_list(games, seed)
        )
    )
    config = EvaluationConfig(
        seeds=evaluation_seeds,
        max_moves=max_moves,
    )
    agent = HeuristicAgent()
    rows, metadata = evaluate_policy(
        agent,
        config,
        agent_id="heuristic_strategy_score",
        architecture="hand_coded_strategy_score",
    )
    csv_path, json_path = write_results(
        output_dir,
        rows,
        metadata,
    )
    print(f"Per-game results saved to {csv_path}")
    print(f"Summary and metadata saved to {json_path}")
    return rows, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="results/canonical/heuristic"
    )
    args = parser.parse_args()
    explicit_seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else None
    )
    evaluate(
        games=args.games,
        seed=args.seed,
        max_moves=args.max_moves,
        seeds=explicit_seeds,
        output_dir=args.output_dir,
    )
