#!/usr/bin/env python3
"""Compare encode_board vs encode_board_v2 training results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game2048.agents.apprentice_agent import (
    ApprenticeAgent,
    encode_board,
    encode_board_v2,
)
from game2048.agents.expectimax_agent import ExpectimaxAgent
from game2048.game import Game2048


def evaluate_agent(agent: ApprenticeAgent, num_games: int = 10) -> dict[str, float]:
    total_score = 0
    max_tiles: list[int] = []
    inference_times: list[float] = []
    for seed in range(1, num_games + 1):
        game = Game2048(seed=seed)
        agent.inference_times = []
        score = agent.play_game(game)
        total_score += score
        max_tiles.append(game.get_max_tile())
        if agent.inference_times:
            inference_times.extend(agent.inference_times)
    avg_score = total_score / num_games
    avg_inference = (
        sum(inference_times) / len(inference_times) if inference_times else 0
    )
    return {
        "avg_score": avg_score,
        "avg_max_tile": sum(max_tiles) / len(max_tiles),
        "max_tile": max(max_tiles),
        "avg_inference_ms": avg_inference,
    }


def main() -> int:
    teacher = ExpectimaxAgent(depth=2)
    num_samples = 20000
    epochs = 30
    num_eval_games = 20

    print("=" * 60)
    print("Training with encode_board (v1) - 64 features")
    print("=" * 60)
    agent_v1 = ApprenticeAgent.from_training(
        num_samples=num_samples,
        teacher=teacher,
        hidden_sizes=[128, 128, 128],
        epochs=epochs,
        lr=0.001,
        batch_size=64,
        verbose=True,
        seed=42,
        patience=5,
        save_logs=False,
        encoding_version=1,
    )
    results_v1 = evaluate_agent(agent_v1, num_games=num_eval_games)
    print(f"\nV1 Results ({num_eval_games} games):")
    print(f"  Average Score: {results_v1['avg_score']:.0f}")
    print(f"  Average Max Tile: {results_v1['avg_max_tile']:.0f}")
    print(f"  Best Max Tile: {results_v1['max_tile']}")
    print(f"  Avg Inference Time: {results_v1['avg_inference_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("Training with encode_board_v2 - 26 features")
    print("=" * 60)
    agent_v2 = ApprenticeAgent.from_training(
        num_samples=num_samples,
        teacher=teacher,
        hidden_sizes=[128, 128, 128],
        epochs=epochs,
        lr=0.001,
        batch_size=64,
        verbose=True,
        seed=42,
        patience=5,
        save_logs=False,
        encoding_version=2,
    )
    results_v2 = evaluate_agent(agent_v2, num_games=num_eval_games)
    print(f"\nV2 Results ({num_eval_games} games):")
    print(f"  Average Score: {results_v2['avg_score']:.0f}")
    print(f"  Average Max Tile: {results_v2['avg_max_tile']:.0f}")
    print(f"  Best Max Tile: {results_v2['max_tile']}")
    print(f"  Avg Inference Time: {results_v2['avg_inference_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    improvement = (
        (results_v2["avg_score"] - results_v1["avg_score"])
        / results_v1["avg_score"]
        * 100
    )
    print(f"V1 (64 features): {results_v1['avg_score']:.0f} avg score")
    print(f"V2 (26 features): {results_v2['avg_score']:.0f} avg score")
    print(f"Improvement: {improvement:+.1f}%")

    if improvement > 5:
        print("\nV2 shows >5% improvement. Recommend using encode_board_v2 as default.")
        return 0
    elif results_v2["avg_score"] > 2500:
        print("\nV2 meets target >2500 avg score.")
        return 0
    else:
        print("\nV2 did not meet improvement threshold.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
