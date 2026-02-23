#!/usr/bin/env python3
"""Benchmark 2048 agents across multiple seeds and report statistics."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from game2048.stats import compute_all_statistics, format_rank_range, get_rank_emoji


def get_agent_class(name: str) -> Optional[type]:
    """Get agent class by name."""
    agents = {
        "random": RandomAgent,
        "rightleft": RightLeftAgent,
        "rightdown": RightDownAgent,
        "corner": CornerAgent,
        "greedy": GreedyAgent,
        "snake": SnakeAgent,
        "expectimax": ExpectimaxAgent,
        "mcts": MCTSAgent,
        "td_learning": TDLearningAgent,
    }
    return agents.get(name)


def create_agent(name: str) -> BaseAgent:
    """Create an agent instance by name."""
    agent_classes = {
        "random": lambda: RandomAgent(seed=42),
        "rightleft": lambda: RightLeftAgent(),
        "rightdown": lambda: RightDownAgent(),
        "corner": lambda: CornerAgent(),
        "greedy": lambda: GreedyAgent(),
        "snake": lambda: SnakeAgent(),
        "expectimax": lambda: ExpectimaxAgent(depth=2),
        "mcts": lambda: MCTSAgent(simulations=20),
        "td_learning": lambda: TDLearningAgent(seed=42),
    }
    return agent_classes[name]()


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


def benchmark_agent_worker(agent_name: str, seeds: list[int]) -> tuple[str, list[dict]]:
    """Worker function for parallel benchmarking - creates agent and runs games."""
    agent = create_agent(agent_name)
    results = []
    for seed in seeds:
        result = run_game(agent, seed)
        results.append(result)
    return agent_name, results


def run_parallel_benchmarks(
    agent_names: list[str],
    seeds: range,
    max_workers: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, list[dict]]:
    """Run all agents in parallel using ProcessPoolExecutor."""
    if max_workers is None:
        max_workers = min(len(agent_names), mp.cpu_count())

    seeds_list = list(seeds)
    all_results: dict[str, list[dict]] = {}

    if verbose:
        print(
            f"Running {len(agent_names)} agents in parallel with {max_workers} workers..."
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(benchmark_agent_worker, name, seeds_list): name
            for name in agent_names
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                name, results = future.result()
                all_results[name] = results
                if verbose:
                    print(f"  Completed {name} agent")
            except Exception as e:
                print(f"Error running {agent_name}: {e}")
                all_results[agent_name] = []

    return all_results


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


def print_comparison(all_results: dict[str, list[dict]], alpha: float = 0.05) -> None:
    """Print comparison table of all agents with statistical analysis."""
    from game2048.stats import compute_all_statistics, format_rank_range, get_rank_emoji

    stats_data = compute_all_statistics(all_results, alpha=alpha)
    summaries = stats_data["summaries"]
    confidence_intervals = stats_data["confidence_intervals"]
    rank_ranges = stats_data["rank_ranges"]

    sorted_agents = sorted(
        summaries.keys(), key=lambda a: summaries[a]["avg_score"], reverse=True
    )

    print("\n" + "=" * 80)
    print("COMPARISON (with 95% confidence intervals)")
    print("=" * 80)
    print(
        f"{'Rank':<12} {'Agent':<12} {'Avg Score':>10} {'95% CI':>16} {'Avg Tile':>10}"
    )
    print("-" * 80)

    for name in sorted_agents:
        summary = summaries[name]
        ci = confidence_intervals[name]
        rank_range = rank_ranges[name]
        emoji = get_rank_emoji(rank_range)
        rank_str = f"{emoji} {format_rank_range(rank_range)}".strip()
        ci_str = f"[{ci[0]:.0f}, {ci[1]:.0f}]"
        print(
            f"{rank_str:<12} {name:<12} {summary['avg_score']:>10.1f} {ci_str:>16} {summary['avg_max_tile']:>10.1f}"
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
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run agents in parallel using multiple processes",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on CPU count)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution (default behavior)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
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
    use_parallel = args.parallel or args.jobs is not None
    if args.sequential:
        use_parallel = False

    if not args.json:
        print(f"Benchmarking 2048 agents with seeds 1-{args.seeds}...")
        if use_parallel:
            workers = args.jobs or min(len(agents), mp.cpu_count())
            print(f"Using parallel execution with {workers} workers")
        print("=" * 60)

    if use_parallel:
        agent_names = list(agents.keys())
        all_results = run_parallel_benchmarks(
            agent_names, seeds, max_workers=args.jobs, verbose=not args.json
        )
    else:
        for name, agent in agents.items():
            if not args.json:
                print(f"\nRunning {name} agent...")
            results = benchmark_agent(
                agent, seeds, verbose=args.verbose and not args.json
            )
            all_results[name] = results

    if args.json:
        stats_data = compute_all_statistics(all_results, alpha=args.alpha)
        output_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agents": all_results,
            "summary": stats_data["summaries"],
            "confidence_intervals": {
                k: {"lower": v[0], "upper": v[1]}
                for k, v in stats_data["confidence_intervals"].items()
            },
            "rank_ranges": {
                k: {"min": v[0], "max": v[1], "display": format_rank_range(v)}
                for k, v in stats_data["rank_ranges"].items()
            },
            "statistical_params": {
                "alpha": args.alpha,
                "confidence": 0.95,
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
        print_comparison(all_results, alpha=args.alpha)


if __name__ == "__main__":
    main()
