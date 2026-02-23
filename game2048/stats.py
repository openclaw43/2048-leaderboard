"""Statistical analysis utilities for benchmark results."""

import numpy as np
from scipy import stats


def confidence_interval(
    scores: list[float] | list[int], confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate confidence interval for a set of scores.

    Args:
        scores: List of score values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(scores) < 2:
        mean = np.mean(scores) if scores else 0
        return (mean, mean)

    scores_arr = np.array(scores)
    mean = np.mean(scores_arr)
    sem = stats.sem(scores_arr)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    return (mean - margin, mean + margin)


def is_significantly_different(
    scores_a: list[float] | list[int],
    scores_b: list[float] | list[int],
    alpha: float = 0.05,
) -> bool:
    """
    Test if two sets of scores are significantly different using Welch's t-test.

    Welch's t-test is used because it doesn't assume equal variance between groups.

    Args:
        scores_a: First set of scores
        scores_b: Second set of scores
        alpha: Significance level (default 0.05)

    Returns:
        True if the difference is statistically significant
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return False

    t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return p_value < alpha


def compute_rank_ranges(
    all_scores: dict[str, list[float] | list[int]], alpha: float = 0.05
) -> dict[str, tuple[int, int]]:
    """
    Compute rank ranges for agents based on statistical significance.

    Groups agents into statistically distinct tiers. If two agents are not
    significantly different, they share a rank range.

    Args:
        all_scores: Dict mapping agent names to their score lists
        alpha: Significance level for determining if agents differ

    Returns:
        Dict mapping agent names to (min_rank, max_rank) tuples.

        Example:
            {
                'mcts': (1, 1),        # Clearly #1
                'expectimax': (2, 3),  # Tied for 2nd-3rd
                'heuristic': (2, 3),   # Tied with expectimax
                'snake': (4, 4),       # Clearly #4
            }
    """
    agents = list(all_scores.keys())
    if not agents:
        return {}

    avg_scores = {agent: np.mean(scores) for agent, scores in all_scores.items()}
    sorted_agents = sorted(agents, key=lambda a: avg_scores[a], reverse=True)

    n = len(sorted_agents)
    significant_diff = {agent: set() for agent in agents}

    for i in range(n):
        for j in range(i + 1, n):
            agent_a = sorted_agents[i]
            agent_b = sorted_agents[j]

            if is_significantly_different(
                all_scores[agent_a], all_scores[agent_b], alpha
            ):
                significant_diff[agent_a].add(agent_b)
                significant_diff[agent_b].add(agent_a)

    rank_ranges = {}
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
    """
    Format a rank range for display.

    Args:
        rank_range: Tuple of (min_rank, max_rank)

    Returns:
        Formatted string like "1st", "2nd-3rd", etc.
    """
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
    """Get emoji for top ranks."""
    min_rank, _ = rank_range
    if min_rank == 1:
        return "🥇"
    elif min_rank == 2:
        return "🥈"
    elif min_rank == 3:
        return "🥉"
    return ""


def compute_all_statistics(
    all_results: dict[str, list[dict]], alpha: float = 0.05, confidence: float = 0.95
) -> dict:
    """
    Compute comprehensive statistics for all agents.

    Args:
        all_results: Dict mapping agent names to list of game result dicts
        alpha: Significance level for rank grouping
        confidence: Confidence level for intervals

    Returns:
        Dict with summary stats, confidence intervals, and rank ranges
    """
    all_scores = {}
    summaries = {}
    confidence_intervals = {}

    for agent_name, results in all_results.items():
        scores = [r["score"] for r in results]
        max_tiles = [r["max_tile"] for r in results]

        all_scores[agent_name] = scores
        summaries[agent_name] = {
            "avg_score": round(float(np.mean(scores)), 1),
            "max_score": int(max(scores)),
            "min_score": int(min(scores)),
            "avg_max_tile": round(float(np.mean(max_tiles)), 1),
        }
        confidence_intervals[agent_name] = confidence_interval(scores, confidence)

    rank_ranges = compute_rank_ranges(all_scores, alpha)

    return {
        "summaries": summaries,
        "confidence_intervals": confidence_intervals,
        "rank_ranges": rank_ranges,
        "alpha": alpha,
        "confidence": confidence,
    }
