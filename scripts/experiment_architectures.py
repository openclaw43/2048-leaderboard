#!/usr/bin/env python3
"""Experiment with different neural network architectures for Apprentice Agent."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game2048.agents.apprentice_agent import (
    ApprenticeAgent,
    generate_training_data,
    train_network,
)
from game2048.agents.expectimax_agent import ExpectimaxAgent
from game2048.game import Game2048

ARCHITECTURES = {
    "wide": [256, 128],
    "deep": [128, 128, 128],
    "wide_deep": [256, 256, 128],
}

TRAIN_SAMPLES = 20000
EPOCHS = 30
LR = 0.001
BATCH_SIZE = 128
SEED = 42
EVAL_GAMES = 10


def measure_inference_time(agent: ApprenticeAgent, num_samples: int = 100) -> float:
    inference_times = []
    for seed in range(1, num_samples + 1):
        game = Game2048(seed=seed)
        while not game.game_over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            start = time.perf_counter()
            agent.choose_move(game)
            elapsed = (time.perf_counter() - start) * 1000
            inference_times.append(elapsed)
            game.move(valid_moves[0])
    if not inference_times:
        return 0.0
    return sum(inference_times) / len(inference_times)


def evaluate_agent(agent: ApprenticeAgent, num_games: int = EVAL_GAMES) -> dict:
    total_score = 0
    max_tiles = []
    for seed in range(1, num_games + 1):
        game = Game2048(seed=seed)
        score = agent.play_game(game)
        total_score += score
        max_tiles.append(game.get_max_tile())
    avg_inference = measure_inference_time(agent)
    return {
        "avg_score": total_score / num_games,
        "max_tile": max(max_tiles),
        "avg_max_tile": sum(max_tiles) / len(max_tiles),
        "avg_inference_ms": avg_inference,
    }


def run_experiment(
    name: str, hidden_sizes: list[int], X, y
) -> tuple[dict, ApprenticeAgent]:
    print(f"\n{'=' * 60}")
    print(f"Architecture: {name} (64→{'→'.join(map(str, hidden_sizes))}→4)")
    print(f"{'=' * 60}")

    start_time = time.time()
    agent = ApprenticeAgent(hidden_sizes=hidden_sizes)
    agent.network, history = train_network(
        X,
        y,
        hidden_sizes=hidden_sizes,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        verbose=True,
        seed=SEED,
    )
    train_time = time.time() - start_time

    print(f"\nEvaluating trained model...")
    results = evaluate_agent(agent)
    results["val_accuracy"] = history["val_accuracy"]
    results["train_time_s"] = train_time
    results["hidden_sizes"] = hidden_sizes
    results["name"] = name

    print(f"\nResults for {name}:")
    print(f"  Val Accuracy: {results['val_accuracy']:.2%}")
    print(f"  Avg Score: {results['avg_score']:.0f}")
    print(f"  Avg Max Tile: {results['avg_max_tile']:.0f}")
    print(f"  Avg Inference: {results['avg_inference_ms']:.2f}ms")
    print(f"  Training Time: {train_time:.1f}s")

    return results, agent


def main():
    print("=" * 60)
    print("Architecture Experiment for Apprentice Agent")
    print("=" * 60)
    print(f"Training samples: {TRAIN_SAMPLES}")
    print(f"Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
    print(f"Seed: {SEED}")

    print("\nGenerating training data from Expectimax(depth=2)...")
    teacher = ExpectimaxAgent(depth=2)
    X, y = generate_training_data(
        num_samples=TRAIN_SAMPLES, teacher=teacher, verbose=True, seed=SEED
    )

    results = {}
    agents = {}

    for name, hidden_sizes in ARCHITECTURES.items():
        result, agent = run_experiment(name, hidden_sizes, X, y)
        results[name] = result
        agents[name] = agent

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(
        f"{'Architecture':<15} {'Val Acc':>10} {'Avg Score':>10} {'Inference':>10} {'Train Time':>12}"
    )
    print("-" * 60)

    best_name = None
    best_score = -1
    for name, r in results.items():
        print(
            f"{name:<15} {r['val_accuracy']:>9.1%} {r['avg_score']:>10.0f} "
            f"{r['avg_inference_ms']:>9.2f}ms {r['train_time_s']:>10.1f}s"
        )
        if r["avg_inference_ms"] < 5.0:
            score = r["val_accuracy"] * 100 + r["avg_score"] / 100
            if score > best_score:
                best_score = score
                best_name = name

    print("-" * 60)

    if best_name is None:
        print("WARNING: No architecture met the <5ms inference requirement")
        best_name = min(results.keys(), key=lambda n: results[n]["avg_inference_ms"])

    best = results[best_name]
    print(f"\nBest architecture: {best_name}")
    print(f"  Hidden sizes: {best['hidden_sizes']}")
    print(f"  Val accuracy: {best['val_accuracy']:.2%}")
    print(f"  Avg score: {best['avg_score']:.0f}")
    print(f"  Inference: {best['avg_inference_ms']:.2f}ms")

    accepts = best["val_accuracy"] > 0.45 or best["avg_score"] > 2500
    print(f"\nAcceptance criteria: {'PASSED' if accepts else 'NOT MET'}")
    print(f"  (>45% val accuracy OR >2500 avg score)")

    output_dir = Path(__file__).parent.parent / "game2048" / "agents"
    model_path = output_dir / "apprentice_model.pkl"

    print(f"\nSaving best model to {model_path}...")
    agents[best_name].save_model(str(model_path))
    print("Done!")

    return results


if __name__ == "__main__":
    main()
