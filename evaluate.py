"""Evaluate a trained model with the canonical evaluation framework."""

import argparse
import os

from agent import DQNAgent
from canonical_evaluation import (
    EvaluationConfig,
    evaluate_policy,
    make_seed_list,
    write_results,
)


def evaluate(
    checkpoint_path,
    games=100,
    seed=1000,
    csv_path="results/canonical/games.csv",
    max_moves=5000,
    seeds=None,
):
    """Evaluate one checkpoint and write canonical results."""
    evaluation_seeds = tuple(
        int(value) for value in (
            seeds if seeds is not None else make_seed_list(games, seed)
        )
    )
    config = EvaluationConfig(
        seeds=evaluation_seeds,
        max_moves=max_moves,
    )

    agent = DQNAgent(seed=seed)
    checkpoint = agent.load(checkpoint_path)
    agent.epsilon = 0.0

    def policy(state, valid_actions):
        return agent.act(
            state,
            explore=False,
            valid_actions=valid_actions,
        )

    metadata = checkpoint.get("metadata", {})
    architecture = checkpoint.get(
        "architecture",
        metadata.get("model", "dueling_double_dqn"),
    )
    agent_id = os.path.basename(os.path.normpath(checkpoint_path))
    output_dir = os.path.dirname(csv_path) or "."
    rows, run_metadata = evaluate_policy(
        policy,
        config,
        agent_id=agent_id,
        checkpoint_path=checkpoint_path,
        architecture=architecture,
    )
    csv_output, json_output = write_results(
        output_dir,
        rows,
        run_metadata,
    )
    print(f"Per-game results saved to {csv_output}")
    print(f"Summary and metadata saved to {json_output}")
    return rows, run_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn_2048.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="results/canonical"
    )
    args = parser.parse_args()
    explicit_seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else None
    )
    evaluate(
        args.checkpoint,
        games=args.games,
        seed=args.seed,
        max_moves=args.max_moves,
        seeds=explicit_seeds,
        csv_path=os.path.join(args.output_dir, "games.csv"),
    )
