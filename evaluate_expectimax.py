"""Evaluate an Expectimax 2048 policy with observable canonical output."""

import argparse
import csv
import json
import os

from canonical_evaluation import (
    CSV_FIELDS,
    STATE_REPRESENTATION_VERSION,
    EvaluationConfig,
    _git_revision,
    evaluate_policy,
    make_seed_list,
    summary_from_csv,
)
from expectimax_agent import ExpectimaxAgent


def evaluate(
    depth=2,
    games=100,
    seed=1000,
    seeds=None,
    max_moves=5000,
    output_dir=None,
):
    output_dir = output_dir or (
        f"results/canonical/expectimax_depth_{depth}_seeds"
    )
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
    agent = ExpectimaxAgent(depth=depth)

    os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, "games.csv")
    json_path = os.path.join(output_dir, "summary.json")
    metadata = {
        **agent.metadata(),
        "checkpoint_path": None,
        "git_revision": _git_revision(),
        "state_representation_version": STATE_REPRESENTATION_VERSION,
        "evaluation": config.as_dict(),
    }

    with open(csv_path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
    with open(json_path, "w") as handle:
        json.dump({"metadata": metadata, "summary": None}, handle, indent=2)
        handle.write("\n")

    completed_rows = []
    total_decisions = 0
    total_seconds = 0.0
    total_nodes = 0
    total_cache_hits = 0

    def policy(state, valid_actions, board):
        nonlocal total_decisions, total_seconds
        nonlocal total_nodes, total_cache_hits
        action = agent.select_action(board, valid_actions)
        stats = agent.search_stats()
        total_decisions += 1
        total_seconds += stats["decision_seconds"]
        total_nodes += stats["nodes"]
        total_cache_hits += stats["cache_hits"]
        return action

    def on_game_complete(row):
        completed_rows.append(dict(row))
        with open(csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writerow(row)
            handle.flush()
        summary = summary_from_csv(csv_path)
        with open(json_path, "w") as handle:
            json.dump(
                {"metadata": metadata, "summary": summary},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def on_progress(completed, total):
        average_seconds = (
            total_seconds / total_decisions if total_decisions else 0.0
        )
        latest = completed_rows[-1]
        print(
            f"Game {completed}/{total} | "
            f"score={latest['score']} | "
            f"moves={latest['moves']} | "
            f"avg_decision={average_seconds:.4f}s | "
            f"nodes={total_nodes} | cache_hits={total_cache_hits}",
            flush=True,
        )

    rows, final_metadata = evaluate_policy(
        policy,
        config,
        agent_id="expectimax_strategy_score",
        architecture="expectimax",
        on_game_complete=on_game_complete,
        on_progress=on_progress,
        progress_interval=1,
    )
    metadata.update(final_metadata)
    return rows, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    explicit_seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else None
    )
    evaluate(
        depth=args.depth,
        games=args.games,
        seed=args.seed,
        seeds=explicit_seeds,
        max_moves=args.max_moves,
        output_dir=args.output_dir,
    )
