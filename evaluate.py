"""Evaluate a trained model without exploration."""
import argparse
import csv
import os
import numpy as np
from game import Game2048
from agent import DQNAgent


def evaluate(checkpoint, games=100, seed=1000, csv_path="evaluation.csv"):
    agent = DQNAgent(seed=seed)
    agent.load(checkpoint)
    agent.epsilon = 0.0
    scores, tiles, moves_list = [], [], []
    rows = []
    for i in range(games):
        env = Game2048(seed=seed + i)
        state = env.state_vector()
        done, moves = False, 0
        while not done and moves < 5000:
            action = agent.act(state, explore=False, valid_actions=env.valid_actions())
            _, _, done, _ = env.step(action)
            state = env.state_vector()
            moves += 1
        scores.append(env.score)
        tiles.append(int(env.board.max()))
        moves_list.append(moves)
        rows.append({"game": i + 1, "score": env.score, "max_tile": int(env.board.max()), "moves": moves})

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "score", "max_tile", "moves"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Games evaluated: {games}")
    print(f"Mean score:      {np.mean(scores):.1f}")
    print(f"Median score:    {np.median(scores):.1f}")
    print(f"Mean max tile:   {np.mean(tiles):.1f}")
    print(f"Reached 2048:    {100*np.mean(np.array(tiles) >= 2048):.1f}%")
    print(f"Reached 1024:    {100*np.mean(np.array(tiles) >= 1024):.1f}%")
    print(f"Mean moves:      {np.mean(moves_list):.1f}")
    print(f"Per-game results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn_2048.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--csv", default="evaluation.csv")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.games, args.seed, args.csv)
