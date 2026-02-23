"""Statistical analysis utilities for benchmark results."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


class SummaryStats(TypedDict):
    avg_score: float
    max_score: int
    min_score: int
    avg_max_tile: float


class StatisticsResult(TypedDict):
    summaries: dict[str, SummaryStats]
    confidence_intervals: dict[str, tuple[float, float]]
    rank_ranges: dict[str, tuple[int, int]]
    alpha: float
    confidence: float


def confidence_interval(
    scores: list[float] | list[int], confidence: float = 0.95
) -> tuple[float, float]:
    if len(scores) < 2:
        mean = float(np.mean(scores)) if scores else 0.0
        return (mean, mean)

    scores_arr = np.array([float(s) for s in scores])
    mean = float(np.mean(scores_arr))
    sem = float(stats.sem(scores_arr))
    margin = sem * float(stats.t.ppf((1 + confidence) / 2, len(scores) - 1))
    return (mean - margin, mean + margin)


def is_significantly_different(
    scores_a: list[float] | list[int],
    scores_b: list[float] | list[int],
    alpha: float = 0.05,
) -> bool:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return False

    result = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    p_value = float(result.pvalue)
    return p_value < alpha


def compute_rank_ranges(
    all_scores: dict[str, list[int]], alpha: float = 0.05
) -> dict[str, tuple[int, int]]:
    agents = list(all_scores.keys())
    if not agents:
        return {}

    avg_scores = {agent: float(np.mean(scores)) for agent, scores in all_scores.items()}
    sorted_agents = sorted(agents, key=lambda a: avg_scores[a], reverse=True)

    n = len(sorted_agents)
    significant_diff: dict[str, set[str]] = {agent: set() for agent in agents}

    for i in range(n):
        for j in range(i + 1, n):
            agent_a = sorted_agents[i]
            agent_b = sorted_agents[j]

            if is_significantly_different(
                all_scores[agent_a], all_scores[agent_b], alpha
            ):
                significant_diff[agent_a].add(agent_b)
                significant_diff[agent_b].add(agent_a)

    rank_ranges: dict[str, tuple[int, int]] = {}
    for i, agent in enumerate(sorted_agents):
        base_rank = i + 1

        min_rank = base_rank
        for j in range(i - 1, -1, -1):
            other = sorted_agents[j]
            if other not in significant_diff[agent]:
                min_rank = j + 1
            else:
                break

        max_rank = base_rank
        for j in range(i + 1, n):
            other = sorted_agents[j]
            if other not in significant_diff[agent]:
                max_rank = j + 1
            else:
                break

        rank_ranges[agent] = (min_rank, max_rank)

    return rank_ranges


def format_rank_range(rank_range: tuple[int, int]) -> str:
    min_rank, max_rank = rank_range

    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    if min_rank == max_rank:
        return ordinal(min_rank)
    return f"{ordinal(min_rank)}-{ordinal(max_rank)}"


def get_rank_emoji(rank_range: tuple[int, int]) -> str:
    min_rank, _ = rank_range
    if min_rank == 1:
        return "🥇"
    elif min_rank == 2:
        return "🥈"
    elif min_rank == 3:
        return "🥉"
    return ""


def compute_all_statistics(
    all_results: dict[str, list[dict[str, int]]],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> StatisticsResult:
    all_scores: dict[str, list[int]] = {}
    summaries: dict[str, SummaryStats] = {}
    confidence_intervals: dict[str, tuple[float, float]] = {}

    for agent_name, results in all_results.items():
        scores = [r["score"] for r in results]
        max_tiles = [r["max_tile"] for r in results]

        all_scores[agent_name] = scores
        summaries[agent_name] = SummaryStats(
            avg_score=round(float(np.mean(scores)), 1),
            max_score=int(max(scores)),
            min_score=int(min(scores)),
            avg_max_tile=round(float(np.mean(max_tiles)), 1),
        )
        confidence_intervals[agent_name] = confidence_interval(scores, confidence)

    rank_ranges = compute_rank_ranges(all_scores, alpha)

    return StatisticsResult(
        summaries=summaries,
        confidence_intervals=confidence_intervals,
        rank_ranges=rank_ranges,
        alpha=alpha,
        confidence=confidence,
    )
