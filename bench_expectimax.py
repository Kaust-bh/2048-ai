"""Short deterministic Expectimax search diagnostic."""

import argparse
import time

import numpy as np

from expectimax_agent import ExpectimaxAgent


BOARDS = {
    "sparse": np.array(
        [[2, 2, 4, 8], [16, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ),
    "mid": np.array(
        [[2, 4, 8, 16], [32, 64, 128, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ),
    "dense": np.array(
        [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 0, 0], [0, 0, 0, 0]],
        dtype=np.int64,
    ),
    "terminal": np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]],
        dtype=np.int64,
    ),
}


def run(depths=(1, 2), decisions=1):
    for depth in depths:
        for name, board in BOARDS.items():
            agent = ExpectimaxAgent(depth=depth)
            elapsed = []
            stats = None
            for _ in range(decisions):
                started = time.perf_counter()
                agent.select_action(board)
                elapsed.append(time.perf_counter() - started)
                stats = agent.search_stats()
            print(
                f"board={name} depth={depth} decisions={decisions} "
                f"nodes={stats['nodes']} cache_hits={stats['cache_hits']} "
                f"player_nodes={stats['player_nodes']} "
                f"chance_nodes={stats['chance_nodes']} "
                f"leaves={stats['leaf_evaluations']} "
                f"avg_seconds={np.mean(elapsed):.6f} "
                f"max_seconds={max(elapsed):.6f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default="1")
    parser.add_argument("--decisions", type=int, default=1)
    args = parser.parse_args()
    depths = tuple(int(value) for value in args.depths.split(","))
    run(depths=depths, decisions=args.decisions)
