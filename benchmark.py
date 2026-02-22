#!/usr/bin/env python3
"""Benchmark 2048 agents across multiple seeds and report statistics."""

import argparse
import json
from datetime import datetime
from typing import Optional

from game2048.game import Game2048
from game2048.agents import (
    RandomAgent,
    RightLeftAgent,
    RightDownAgent,
    CornerAgent,
    GreedyAgent,
    SnakeAgent,
    ExpectimaxAgent,
    MCTSAgent,
    TDLearningAgent,
    BaseAgent,
)
from game2048.runner import GameRunner


def run_game(agent: BaseAgent, seed: int) -> dict:
    """Run a single game with given agent and seed, return results."""
    game = Game2048(seed=seed)
    runner = GameRunner(game, agent.choose_move)
    runner.run()
    results = runner.get_results()
    return {
        "seed": seed,
        "score": results["score"],
        "max_tile": results["max_tile"],
        "moves": results["moves"],
    }


def benchmark_agent(
    agent: BaseAgent, seeds: range, verbose: bool = False
) -> list[dict]:
    """Run benchmark for a single agent across all seeds."""
    results = []
    for seed in seeds:
        result = run_game(agent, seed)
        results.append(result)
        if verbose:
            print(
                f"  Seed {seed:2d}: Score={result['score']:5d}, Max={result['max_tile']:4d}, Moves={result['moves']:3d}"
            )
    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics for a set of results."""
    scores = [r["score"] for r in results]
    max_tiles = [r["max_tile"] for r in results]
    return {
        "avg_score": round(sum(scores) / len(scores), 1),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_max_tile": round(sum(max_tiles) / len(max_tiles), 1),
    }


def print_summary(name: str, results: list[dict]) -> None:
    """Print summary statistics for an agent."""
    summary = compute_summary(results)
    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"  Avg Score:  {summary['avg_score']:.1f}")
    print(f"  Max Score:  {summary['max_score']}")
    print(f"  Min Score:  {summary['min_score']}")
    print(f"  Avg Tile:   {summary['avg_max_tile']:.1f}")


def print_comparison(all_results: dict[str, list[dict]]) -> None:
    """Print comparison table of all agents."""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Agent':<12} {'Avg Score':>10} {'Max Score':>10} {'Avg Tile':>10}")
    print("-" * 60)
    for name, results in all_results.items():
        summary = compute_summary(results)
        print(
            f"{name:<12} {summary['avg_score']:>10.1f} {summary['max_score']:>10} {summary['avg_max_tile']:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark 2048 agents")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file for JSON results"
    )
    parser.add_argument(
        "--seeds", type=int, default=50, help="Number of seeds to test (default: 50)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-seed results"
    )
    parser.add_argument(
        "--agents",
        "-a",
        type=str,
        default=None,
        help="Comma-separated list of agents to benchmark (e.g., random,greedy,snake). "
        "If not specified, runs all agents.",
    )
    args = parser.parse_args()

    seeds = range(1, args.seeds + 1)

    all_agents = {
        "random": RandomAgent(seed=42),
        "rightleft": RightLeftAgent(),
        "rightdown": RightDownAgent(),
        "corner": CornerAgent(),
        "greedy": GreedyAgent(),
        "snake": SnakeAgent(),
        "expectimax": ExpectimaxAgent(depth=2),
        "mcts": MCTSAgent(simulations=20),
        "td_learning": TDLearningAgent(seed=42),
    }

    if args.agents:
        requested = [a.strip().lower() for a in args.agents.split(",")]
        invalid = [a for a in requested if a not in all_agents]
        if invalid:
            print(f"Error: Unknown agent(s): {', '.join(invalid)}")
            print(f"Available agents: {', '.join(all_agents.keys())}")
            return 1
        agents = {name: all_agents[name] for name in requested}
    else:
        agents = all_agents

    all_results: dict[str, list[dict]] = {}

    if not args.json:
        print(f"Benchmarking 2048 agents with seeds 1-{args.seeds}...")
        print("=" * 60)

    for name, agent in agents.items():
        if not args.json:
            print(f"\nRunning {name} agent...")
        results = benchmark_agent(agent, seeds, verbose=args.verbose and not args.json)
        all_results[name] = results

    if args.json:
        output_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agents": all_results,
            "summary": {
                name: compute_summary(results) for name, results in all_results.items()
            },
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            if not args.json or args.verbose:
                print(f"Results written to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))
    else:
        for name, results in all_results.items():
            print_summary(name, results)
        print_comparison(all_results)


if __name__ == "__main__":
    main()
