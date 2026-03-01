#!/usr/bin/env python3
"""Evaluate and compare LLM agents across multiple models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from game2048.agents import LLMAgent, LLMHistoryAgent
from game2048.game import Game2048
from game2048.runner import GameRunner


def run_single_game(
    agent: LLMAgent | LLMHistoryAgent,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    game = Game2048(seed=seed)
    runner = GameRunner(game, agent.choose_move)
    runner.run()
    results = runner.get_results()

    if verbose:
        print(
            f"  Seed {seed}: Score={results['score']}, "
            f"Max={results['max_tile']}, Moves={results['moves']}"
        )

    stats = agent.get_stats()

    return {
        "seed": seed,
        "score": results["score"],
        "max_tile": results["max_tile"],
        "moves": results["moves"],
        "total_cost": stats["total_cost"],
        "total_latency": stats["total_latency"],
        "avg_latency": stats["avg_latency"],
        "move_count": stats["move_count"],
    }


def benchmark_model(
    model: str,
    num_games: int,
    agent_type: str,
    start_seed: int,
    api_key: str,
    verbose: bool = False,
    history_size: int = 5,
) -> dict[str, Any]:
    print(f"\nBenchmarking {model} ({agent_type})...")
    print("-" * 60)

    games_results: list[dict[str, Any]] = []

    for i in range(num_games):
        seed = start_seed + i

        if agent_type == "V1":
            agent: LLMAgent | LLMHistoryAgent = LLMAgent(model=model, api_key=api_key)
        else:
            agent = LLMHistoryAgent(
                model=model, api_key=api_key, history_size=history_size
            )

        result = run_single_game(agent, seed, verbose=verbose)
        games_results.append(result)

    scores = [r["score"] for r in games_results]
    costs = [r["total_cost"] for r in games_results]
    latencies = [r["avg_latency"] for r in games_results]
    moves = [r["moves"] for r in games_results]
    max_tiles = [r["max_tile"] for r in games_results]

    avg_score = sum(scores) / len(scores)
    avg_cost = sum(costs) / len(costs)
    avg_latency = sum(latencies) / len(latencies)
    avg_moves = sum(moves) / len(moves)

    total_moves_all = sum(r["move_count"] for r in games_results)
    total_cost_all = sum(r["total_cost"] for r in games_results)

    return {
        "model": model,
        "agent_type": agent_type,
        "num_games": num_games,
        "avg_score": avg_score,
        "median_score": sorted(scores)[len(scores) // 2],
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "avg_moves": avg_moves,
        "total_cost": total_cost_all,
        "total_moves": total_moves_all,
        "cost_per_game": total_cost_all / num_games,
        "scores": scores,
        "max_tiles": max_tiles,
        "games": games_results,
    }


def generate_comparison_table(results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# LLM Model Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Performance Comparison")
    lines.append("")
    lines.append(
        "| Model | Agent | Avg Score | Median Score | Avg Latency (s) | Cost/Game ($) | Avg Moves |"
    )
    lines.append(
        "|-------|-------|-----------|--------------|-----------------|---------------|-----------|"
    )

    for result in results:
        model_short = result["model"].split("/")[-1]
        lines.append(
            f"| {model_short} | {result['agent_type']} | "
            f"{result['avg_score']:.1f} | {result['median_score']} | "
            f"{result['avg_latency']:.3f} | "
            f"${result['cost_per_game']:.4f} | {result['avg_moves']:.1f} |"
        )

    lines.append("")
    lines.append("## Detailed Statistics")
    lines.append("")

    v1_results = [r for r in results if r["agent_type"] == "V1"]
    v2_results = [r for r in results if r["agent_type"] == "V2"]

    if v1_results:
        lines.append("### V1 Agent (Single State)")
        lines.append("")
        for result in v1_results:
            lines.append(f"**{result['model']}**")
            lines.append(f"- Average Score: {result['avg_score']:.1f}")
            lines.append(
                f"- Score Range: {result['min_score']} - {result['max_score']}"
            )
            lines.append(f"- Average Latency: {result['avg_latency']:.3f}s per move")
            lines.append(f"- Average Cost: ${result['cost_per_game']:.4f} per game")
            lines.append("")

    if v2_results:
        lines.append("### V2 Agent (With History)")
        lines.append("")
        for result in v2_results:
            lines.append(f"**{result['model']}**")
            lines.append(f"- Average Score: {result['avg_score']:.1f}")
            lines.append(
                f"- Score Range: {result['min_score']} - {result['max_score']}"
            )
            lines.append(f"- Average Latency: {result['avg_latency']:.3f}s per move")
            lines.append(f"- Average Cost: ${result['cost_per_game']:.4f} per game")
            lines.append("")

    lines.append("## Cost Analysis")
    lines.append("")
    lines.append("| Model | V1 Cost/Game | V2 Cost/Game | Cost Difference |")
    lines.append("|-------|--------------|--------------|-----------------|")

    models = sorted(set(r["model"] for r in results))
    for model in models:
        v1 = next((r for r in v1_results if r["model"] == model), None)
        v2 = next((r for r in v2_results if r["model"] == model), None)

        model_short = model.split("/")[-1]
        v1_cost = f"${v1['cost_per_game']:.4f}" if v1 else "N/A"
        v2_cost = f"${v2['cost_per_game']:.4f}" if v2 else "N/A"

        if v1 and v2:
            diff = v2["cost_per_game"] - v1["cost_per_game"]
            diff_str = f"+${diff:.4f}" if diff >= 0 else f"-${abs(diff):.4f}"
        else:
            diff_str = "N/A"

        lines.append(f"| {model_short} | {v1_cost} | {v2_cost} | {diff_str} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")

    if v1_results:
        best_v1 = max(v1_results, key=lambda r: r["avg_score"])
        lines.append(
            f"- **Best V1 Score**: {best_v1['model']} ({best_v1['avg_score']:.1f})"
        )

    if v2_results:
        best_v2 = max(v2_results, key=lambda r: r["avg_score"])
        lines.append(
            f"- **Best V2 Score**: {best_v2['model']} ({best_v2['avg_score']:.1f})"
        )

    if v1_results and v2_results:
        cheapest_v1 = min(v1_results, key=lambda r: r["cost_per_game"])
        cheapest_v2 = min(v2_results, key=lambda r: r["cost_per_game"])
        lines.append(
            f"- **Cheapest V1**: {cheapest_v1['model']} "
            f"(${cheapest_v1['cost_per_game']:.4f}/game)"
        )
        lines.append(
            f"- **Cheapest V2**: {cheapest_v2['model']} "
            f"(${cheapest_v2['cost_per_game']:.4f}/game)"
        )

    lines.append("")
    lines.append(f"Total games played: {sum(r['num_games'] for r in results)}")
    lines.append(f"Total cost: ${sum(r['total_cost'] for r in results):.4f}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare LLM agents across multiple models"
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        required=True,
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=10,
        help="Number of games per model (default: 10)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="Starting seed (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-game results",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".llm_evaluation_results",
        help="Output directory for results (default: .llm_evaluation_results)",
    )
    parser.add_argument(
        "--v1-only",
        action="store_true",
        help="Only test V1 agent (single state)",
    )
    parser.add_argument(
        "--v2-only",
        action="store_true",
        help="Only test V2 agent (with history)",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=5,
        help="History size for V2 agent (default: 5)",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        print(
            "Run with: uv run --env-file ~/.env.openrouter "
            "python scripts/evaluate_llm_models.py"
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("LLM MODEL EVALUATION")
    print("=" * 60)
    print(f"Models to test: {len(args.models)}")
    for model in args.models:
        print(f"  - {model}")
    print(f"Games per model: {args.games}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    all_results: list[dict[str, Any]] = []

    test_v1 = not args.v2_only
    test_v2 = not args.v1_only

    for model in args.models:
        if test_v1:
            v1_result = benchmark_model(
                model=model,
                num_games=args.games,
                agent_type="V1",
                start_seed=args.start_seed,
                api_key=api_key,
                verbose=args.verbose,
            )
            all_results.append(v1_result)

        if test_v2:
            v2_result = benchmark_model(
                model=model,
                num_games=args.games,
                agent_type="V2",
                start_seed=args.start_seed + args.games,
                api_key=api_key,
                verbose=args.verbose,
                history_size=args.history_size,
            )
            all_results.append(v2_result)

    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)

    raw_results_file = output_dir / f"raw_results_{timestamp}.json"
    with open(raw_results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to: {raw_results_file}")

    comparison_table = generate_comparison_table(all_results)
    comparison_file = output_dir / f"comparison_{timestamp}.md"
    with open(comparison_file, "w") as f:
        f.write(comparison_table)
    print(f"Comparison table saved to: {comparison_file}")

    print("\n" + comparison_table)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
