
"""Train Dueling Double DQN on 2048."""

import argparse
import csv
import os
import random

import numpy as np
import torch

from game import Game2048
from agent import DQNAgent


def evaluate_agent(agent, games=50, seed=10_000, max_moves=5000):
    """Evaluate without exploration on reproducible game seeds."""
    scores = []
    max_tiles = []
    move_counts = []
    rows = []

    for game_index in range(games):
        env = Game2048(seed=seed + game_index)
        state = env.state_vector()
        done = False
        moves = 0

        while not done and moves < max_moves:
            valid_actions = env.valid_actions()

            action = agent.act(
                state,
                explore=False,
                valid_actions=valid_actions,
            )

            _, _, done, _ = env.step(action)
            state = env.state_vector()
            moves += 1

        score = int(env.score)
        max_tile = int(env.board.max())

        scores.append(score)
        max_tiles.append(max_tile)
        move_counts.append(moves)

        rows.append({
            "game": game_index + 1,
            "seed": seed + game_index,
            "score": score,
            "max_tile": max_tile,
            "moves": moves,
        })

    metrics = {
        "games": games,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "mean_max_tile": float(np.mean(max_tiles)),
        "reached_512_pct": 100.0 * sum(
            t >= 512 for t in max_tiles
        ) / games,
        "reached_1024_pct": 100.0 * sum(
            t >= 1024 for t in max_tiles
        ) / games,
        "reached_2048_pct": 100.0 * sum(
            t >= 2048 for t in max_tiles
        ) / games,
        "mean_moves": float(np.mean(move_counts)),
    }

    return metrics, rows


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Game2048(seed=args.seed)

    agent = DQNAgent(
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        target_update=args.target_update,
        seed=args.seed,
    )

    start_episode = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.resume}"
            )

        checkpoint = agent.load(
            args.resume,
            load_optimizer=True,
        )
        start_episode = int(checkpoint.get("episode", 0))

        print(
            f"Resumed at episode {start_episode}. "
            "Replay buffer starts empty."
        )

    log_path = os.path.join(args.out, "training.csv")
    checkpoint_path = os.path.join(
        args.out, "dqn_2048.pt"
    )
    best_path = os.path.join(
        args.out, "best_dqn_2048.pt"
    )
    eval_log_path = os.path.join(
        args.out, "evaluation.csv"
    )

    resume_log = bool(
        args.resume and os.path.exists(log_path)
    )
    best_score = float("-inf")

    if args.resume and os.path.exists(best_path):
        saved = torch.load(
            best_path, map_location="cpu", weights_only=False
        )
        best_score = float(
            saved.get("best_eval_mean_score", float("-inf"))
        )

    mode = "a" if resume_log else "w"

    with open(log_path, mode, newline="") as f:
        fields = [
            "episode", "score", "max_tile",
            "moves", "epsilon", "mean_loss",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)

        if not resume_log:
            writer.writeheader()

        end_episode = start_episode + args.episodes

        for episode in range(
            start_episode + 1, end_episode + 1
        ):
            env.reset()
            state = env.state_vector()
            done = False
            moves = 0
            losses = []

            while not done and moves < args.max_moves:
                valid_actions = env.valid_actions()

                action = agent.act(
                    state,
                    explore=True,
                    valid_actions=valid_actions,
                )

                _, reward, done, _ = env.step(action)
                next_state = env.state_vector()

                next_valid_actions = (
                    [] if done else env.valid_actions()
                )

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_valid_actions,
                )

                loss = agent.learn()

                if loss is not None:
                    losses.append(loss)

                state = next_state
                moves += 1

            agent.decay_epsilon()

            row = {
                "episode": episode,
                "score": env.score,
                "max_tile": int(env.board.max()),
                "moves": moves,
                "epsilon": round(agent.epsilon, 5),
                "mean_loss": (
                    round(float(np.mean(losses)), 6)
                    if losses else ""
                ),
            }

            writer.writerow(row)
            f.flush()

            if (
                episode % args.log_every == 0
                or episode == start_episode + 1
            ):
                print(
                    f"Episode {episode:6d}/{end_episode} | "
                    f"score={env.score:6d} | "
                    f"max tile={int(env.board.max()):5d} | "
                    f"moves={moves:4d} | "
                    f"epsilon={agent.epsilon:.3f} | "
                    f"loss={row['mean_loss']}"
                )

            if (
                episode % args.save_every == 0
                or episode == end_episode
            ):
                agent.save(
                    checkpoint_path,
                    episode=episode,
                    metadata={
                        "args": vars(args),
                        "model": "dueling_double_dqn",
                    },
                )
                print(f"Saved {checkpoint_path}")

            should_evaluate = (
                episode % args.eval_every == 0
                or episode == end_episode
            )

            if should_evaluate:
                metrics, eval_rows = evaluate_agent(
                    agent,
                    games=args.eval_games,
                    seed=args.eval_seed,
                    max_moves=args.max_moves,
                )

                print(
                    f"[Evaluation @ {episode}] "
                    f"mean={metrics['mean_score']:.1f} | "
                    f"median={metrics['median_score']:.1f} | "
                    f"mean max tile={metrics['mean_max_tile']:.1f} | "
                    f"512+={metrics['reached_512_pct']:.1f}% | "
                    f"1024+={metrics['reached_1024_pct']:.1f}% | "
                    f"2048={metrics['reached_2048_pct']:.1f}%"
                )

                write_csv(
                    os.path.join(
                        args.out,
                        f"eval_games_ep{episode}.csv",
                    ),
                    eval_rows,
                )

                eval_exists = os.path.exists(eval_log_path)

                with open(
                    eval_log_path, "a", newline=""
                ) as ef:
                    eval_fields = [
                        "episode", *metrics.keys()
                    ]
                    eval_writer = csv.DictWriter(
                        ef, fieldnames=eval_fields
                    )

                    if not eval_exists:
                        eval_writer.writeheader()

                    eval_writer.writerow({
                        "episode": episode,
                        **metrics,
                    })

                if metrics["mean_score"] > best_score:
                    best_score = metrics["mean_score"]

                    agent.save(
                        best_path,
                        episode=episode,
                        metadata={
                            "args": vars(args),
                            "model": "dueling_double_dqn",
                            "best_eval_mean_score": best_score,
                            "best_eval_metrics": metrics,
                        },
                    )

                    # Store the best score at the top level too.
                    saved = torch.load(
                        best_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    saved["best_eval_mean_score"] = best_score
                    torch.save(saved, best_path)

                    print(
                        f"New best checkpoint: {best_path} "
                        f"(mean score={best_score:.1f})"
                    )

    print("Training finished.")
    print(f"Latest model: {checkpoint_path}")
    print(f"Best evaluated model: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="checkpoints_dueling"
    )
    parser.add_argument("--resume", default="")

    train(parser.parse_args())
