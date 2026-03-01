#!/usr/bin/env python3
"""Benchmark LLM agent for 2048."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game2048.agents import LLMAgent, LLMHistoryAgent
from game2048.game import Game2048
from game2048.runner import GameRunner


def run_game(
    agent: LLMAgent | LLMHistoryAgent, seed: int, verbose: bool = False
) -> dict[str, int]:
    game = Game2048(seed=seed)
    runner = GameRunner(game, agent.choose_move)
    runner.run()
    results = runner.get_results()

    if verbose:
        print(
            f"  Seed {seed}: Score={results['score']}, Max={results['max_tile']}, Moves={results['moves']}"
        )

    return {
        "seed": seed,
        "score": results["score"],
        "max_tile": results["max_tile"],
        "moves": results["moves"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LLM agent for 2048")
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen/qwen3.5-flash-02-23",
        help="Model to use",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-game results"
    )
    parser.add_argument(
        "--start-seed", type=int, default=1, help="Starting seed (default: 1)"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Use LLMHistoryAgent with game history tracking",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=5,
        help="History size for LLMHistoryAgent (default: 5)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        print(
            "Run with: uv run --env-file ~/.env.openrouter python scripts/benchmark_llm.py"
        )
        return 1

    agent_type = "LLMHistoryAgent" if args.history else "LLMAgent"
    print(f"Benchmarking {agent_type} with model: {args.model}")
    if args.history:
        print(f"History size: {args.history_size}")
    print(f"Running {args.games} games...")
    print("=" * 60)

    results: list[dict[str, int]] = []
    total_cost = 0.0
    total_latency = 0.0
    total_moves = 0

    for i in range(args.games):
        seed = args.start_seed + i
        agent: LLMAgent | LLMHistoryAgent
        if args.history:
            agent = LLMHistoryAgent(
                model=args.model, api_key=api_key, history_size=args.history_size
            )
        else:
            agent = LLMAgent(model=args.model, api_key=api_key)
        result = run_game(agent, seed, verbose=args.verbose)
        results.append(result)

        stats = agent.get_stats()
        total_cost += stats["total_cost"]
        total_latency += stats["total_latency"]
        total_moves += int(stats["move_count"])

    scores = [r["score"] for r in results]
    max_tiles = [r["max_tile"] for r in results]
    moves = [r["moves"] for r in results]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Games played:     {args.games}")
    print(f"Average score:    {sum(scores) / len(scores):.1f}")
    print(f"Median score:     {sorted(scores)[len(scores) // 2]}")
    print(f"Max score:        {max(scores)}")
    print(f"Min score:        {min(scores)}")
    print(f"Average moves:    {sum(moves) / len(moves):.1f}")
    print(f"Average max tile: {sum(max_tiles) / len(max_tiles):.1f}")
    print(f"\nLatency & Cost:")
    print(
        f"  Avg latency per move: {total_latency / total_moves:.3f}s"
        if total_moves > 0
        else "  Avg latency per move: N/A"
    )
    print(f"  Total cost:           ${total_cost:.4f}")
    print(f"  Avg cost per game:    ${total_cost / args.games:.4f}")

    print(f"\nMax tiles reached:")
    tile_counts: dict[int, int] = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    for tile in sorted(tile_counts.keys(), reverse=True):
        print(
            f"  {tile}: {tile_counts[tile]} games ({tile_counts[tile] / args.games * 100:.0f}%)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
