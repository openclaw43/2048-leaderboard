#!/usr/bin/env python3
"""Train an Apprentice agent to imitate MCTS behavior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game2048.agents.apprentice_agent import ApprenticeAgent
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
    parser = argparse.ArgumentParser(description="Train Apprentice agent")
    parser.add_argument(
        "--samples", type=int, default=20000, help="Number of training samples"
    )
    parser.add_argument(
        "--mcts-simulations", type=int, default=20, help="MCTS simulations for teacher"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--hidden-sizes", type=str, default="128,64", help="Hidden layer sizes"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output model path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-games", type=int, default=10, help="Number of evaluation games"
    )
    args = parser.parse_args()
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    print(f"Training Apprentice agent with {args.samples} samples...")
    print(f"Hidden sizes: {hidden_sizes}, Epochs: {args.epochs}, LR: {args.lr}")
    agent = ApprenticeAgent.from_training(
        num_samples=args.samples,
        mcts_simulations=args.mcts_simulations,
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=True,
        seed=args.seed,
    )
    print("\nEvaluating trained agent...")
    results = evaluate_agent(agent, num_games=args.eval_games)
    print(f"\nEvaluation Results ({args.eval_games} games):")
    print(f"  Average Score: {results['avg_score']:.0f}")
    print(f"  Average Max Tile: {results['avg_max_tile']:.0f}")
    print(f"  Best Max Tile: {results['max_tile']}")
    print(f"  Avg Inference Time: {results['avg_inference_ms']:.2f}ms")
    output_path = args.output or Path(__file__).parent / "apprentice_model.pkl"
    agent.save_model(str(output_path))
    print(f"\nModel saved to: {output_path}")
    if results["avg_score"] < 15000:
        print("\nWarning: Average score < 15k. Consider more training data or epochs.")
    if results["avg_inference_ms"] > 10:
        print("\nWarning: Inference time > 10ms. Consider smaller hidden layers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
