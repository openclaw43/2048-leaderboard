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


class DistributionMetrics(TypedDict):
    median: float
    std_dev: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    percentile_99: float
    iqr: float


class WinRates(TypedDict):
    reached_512: float
    reached_1024: float
    reached_2048: float


class GameLengthStats(TypedDict):
    avg_moves: float
    min_moves: int
    max_moves: int
    median_moves: float


class ConsistencyScore(TypedDict):
    coefficient_of_variation: float


class ExtendedStats(TypedDict):
    distribution: DistributionMetrics
    win_rates: WinRates
    game_length: GameLengthStats
    consistency: ConsistencyScore


class StatisticsResult(TypedDict):
    summaries: dict[str, SummaryStats]
    confidence_intervals: dict[str, tuple[float, float]]
    rank_ranges: dict[str, tuple[int, int]]
    alpha: float
    confidence: float
    extended: dict[str, ExtendedStats]


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


def compute_distribution_metrics(scores: list[int]) -> DistributionMetrics:
    if not scores:
        return DistributionMetrics(
            median=0.0,
            std_dev=0.0,
            percentile_25=0.0,
            percentile_75=0.0,
            percentile_90=0.0,
            percentile_99=0.0,
            iqr=0.0,
        )
    scores_arr = np.array(scores)
    p25 = float(np.percentile(scores_arr, 25))
    p75 = float(np.percentile(scores_arr, 75))
    return DistributionMetrics(
        median=float(np.median(scores_arr)),
        std_dev=float(np.std(scores_arr, ddof=1)),
        percentile_25=p25,
        percentile_75=p75,
        percentile_90=float(np.percentile(scores_arr, 90)),
        percentile_99=float(np.percentile(scores_arr, 99)),
        iqr=p75 - p25,
    )


def compute_win_rates(results: list[dict[str, int]]) -> WinRates:
    if not results:
        return WinRates(reached_512=0.0, reached_1024=0.0, reached_2048=0.0)
    total = len(results)
    reached_512 = sum(1 for r in results if r["max_tile"] >= 512)
    reached_1024 = sum(1 for r in results if r["max_tile"] >= 1024)
    reached_2048 = sum(1 for r in results if r["max_tile"] >= 2048)
    return WinRates(
        reached_512=round(reached_512 / total * 100, 1),
        reached_1024=round(reached_1024 / total * 100, 1),
        reached_2048=round(reached_2048 / total * 100, 1),
    )


def compute_game_length_stats(results: list[dict[str, int]]) -> GameLengthStats:
    if not results:
        return GameLengthStats(
            avg_moves=0.0, min_moves=0, max_moves=0, median_moves=0.0
        )
    moves = [r["moves"] for r in results]
    return GameLengthStats(
        avg_moves=round(float(np.mean(moves)), 1),
        min_moves=int(min(moves)),
        max_moves=int(max(moves)),
        median_moves=float(np.median(moves)),
    )


def compute_consistency_score(scores: list[int]) -> ConsistencyScore:
    if not scores:
        return ConsistencyScore(coefficient_of_variation=0.0)
    mean = float(np.mean(scores))
    if mean == 0:
        return ConsistencyScore(coefficient_of_variation=0.0)
    std_dev = float(np.std(scores, ddof=1))
    return ConsistencyScore(coefficient_of_variation=round(std_dev / mean * 100, 1))


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
    extended: dict[str, ExtendedStats] = {}

    for agent_name, results in all_results.items():
        scores = [r["score"] for r in results]
        max_tiles = [r["max_tile"] for r in results]

        all_scores[agent_name] = scores
        summaries[agent_name] = SummaryStats(
            avg_score=round(float(np.mean(scores)), 1),
            max_score=int(max(scores)) if scores else 0,
            min_score=int(min(scores)) if scores else 0,
            avg_max_tile=round(float(np.mean(max_tiles)), 1),
        )
        confidence_intervals[agent_name] = confidence_interval(scores, confidence)

        extended[agent_name] = ExtendedStats(
            distribution=compute_distribution_metrics(scores),
            win_rates=compute_win_rates(results),
            game_length=compute_game_length_stats(results),
            consistency=compute_consistency_score(scores),
        )

    rank_ranges = compute_rank_ranges(all_scores, alpha)

    return StatisticsResult(
        summaries=summaries,
        confidence_intervals=confidence_intervals,
        rank_ranges=rank_ranges,
        alpha=alpha,
        confidence=confidence,
        extended=extended,
    )
