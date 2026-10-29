"""Evaluate a legacy Strategy V1 checkpoint canonically."""

import argparse
import os

from canonical_evaluation import (
    EvaluationConfig,
    evaluate_policy,
    make_seed_list,
    write_results,
)
from legacy_agent import LegacyDQNAgent, action_mapping_metadata


def evaluate(
    checkpoint_path="backups/strategy_dqn_v1.pt",
    games=100,
    seed=1000,
    seeds=None,
    max_moves=5000,
    output_dir="results/canonical/neural_strategy_v1_seeds",
):
    """Evaluate a legacy checkpoint without overwriting an output directory."""
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )

    evaluation_seeds = tuple(
        int(value) for value in (
            seeds if seeds is not None else make_seed_list(games, seed)
        )
    )
    config = EvaluationConfig(
        seeds=evaluation_seeds,
        max_moves=max_moves,
    )
    agent = LegacyDQNAgent.load(checkpoint_path)
    rows, metadata = evaluate_policy(
        agent,
        config,
        agent_id="legacy_strategy_v1",
        checkpoint_path=checkpoint_path,
        architecture="legacy_feedforward_dqn",
    )
    metadata.update(action_mapping_metadata())
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
    parser.add_argument(
        "--checkpoint", default="backups/strategy_dqn_v1.pt"
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        default="results/canonical/neural_strategy_v1_seeds",
    )
    args = parser.parse_args()
    explicit_seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else None
    )
    evaluate(
        checkpoint_path=args.checkpoint,
        games=args.games,
        seed=args.seed,
        seeds=explicit_seeds,
        max_moves=args.max_moves,
        output_dir=args.output_dir,
    )
