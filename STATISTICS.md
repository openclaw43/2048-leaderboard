# Statistical Methodology

This document describes the statistical methods used in the 2048 AI Leaderboard to provide confidence intervals, rank ranges, and extended statistics.

## Overview

Instead of showing exact rankings that may be misleading due to random variance, we use statistical significance testing to group agents into performance tiers. We also provide extended statistics including distribution metrics, win rates, game length statistics, and consistency scores.

## Basic Statistics

### Summary Statistics

For each agent, we compute:
- **Average Score**: Mean score across all games
- **Max Score**: Highest score achieved
- **Min Score**: Lowest score achieved
- **Avg Max Tile**: Average maximum tile achieved

## Confidence Intervals

We calculate 95% confidence intervals for the mean score of each agent using the t-distribution:

```python
from scipy import stats

mean = np.mean(scores)
sem = stats.sem(scores)  # Standard error of mean
margin = sem * stats.t.ppf((1 + 0.95) / 2, len(scores) - 1)
ci_lower = mean - margin
ci_upper = mean + margin
```

### Interpretation

- The 95% confidence interval means: if we ran the benchmark many times, 95% of the calculated intervals would contain the true mean score
- Wider intervals indicate more variability in the agent's performance
- Narrower intervals indicate more consistent performance

## Rank Ranges

We use Welch's t-test to determine if agents have statistically significantly different performance:

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
is_different = p_value < alpha  # default alpha = 0.05
```

### Welch's t-test

We use Welch's t-test (unequal variances t-test) because:
- It does not assume equal variance between groups
- Agent scores may have different variances
- It's more robust than Student's t-test for unequal sample sizes

### Rank Grouping Algorithm

1. Sort agents by average score (descending)
2. For each pair of agents, test if they are significantly different
3. Group agents that are NOT significantly different into the same tier
4. Assign rank ranges based on tiers

Example output:
```
{
    'mcts': (1, 1),        # Clearly #1 - significantly better than all others
    'expectimax': (2, 3),  # Not significantly different from heuristic
    'heuristic': (2, 3),   # Not significantly different from expectimax
    'snake': (4, 4),       # Clearly #4 - significantly worse than top 3
}
```

## Extended Statistics

### Distribution Metrics

We compute several distribution metrics to characterize score spread:

- **Median**: The middle value (50th percentile), more robust to outliers than mean
- **Standard Deviation**: Measure of score variability
- **Percentiles**: 25th, 75th, 90th, and 99th percentiles
- **Interquartile Range (IQR)**: 75th - 25th percentile, measuring middle 50% spread

```python
def compute_distribution_metrics(scores):
    return {
        "median": np.median(scores),
        "std_dev": np.std(scores, ddof=1),
        "percentile_25": np.percentile(scores, 25),
        "percentile_75": np.percentile(scores, 75),
        "percentile_90": np.percentile(scores, 90),
        "percentile_99": np.percentile(scores, 99),
        "iqr": np.percentile(scores, 75) - np.percentile(scores, 25),
    }
```

### Win Rates

Win rates measure the percentage of games where an agent achieves a certain tile threshold:

- **reached_512**: % of games reaching 512 tile
- **reached_1024**: % of games reaching 1024 tile
- **reached_2048**: % of games reaching 2048 tile (the "win" condition)

```python
def compute_win_rates(results, thresholds=[512, 1024, 2048]):
    win_rates = {}
    for threshold in thresholds:
        count = sum(1 for r in results if r["max_tile"] >= threshold)
        win_rates[f"reached_{threshold}"] = count / len(results) * 100
    return win_rates
```

### Game Length Statistics

Statistics about how many moves each game takes:

- **Avg Moves**: Average number of moves per game
- **Min Moves**: Fewest moves in any game
- **Max Moves**: Most moves in any game
- **Median Moves**: Median number of moves

Longer games generally indicate better play, as the agent survives longer before the board fills up.

### Consistency Score

The **Coefficient of Variation (CV)** measures consistency as a percentage:

```python
cv = (std_dev / mean) * 100
```

- **Lower CV = More Consistent**: The agent produces similar scores each game
- **Higher CV = Less Consistent**: The agent has high variance in scores

For example:
- CV of 20%: Scores are fairly consistent around the mean
- CV of 50%: Scores vary widely from game to game

## Significance Level (α)

The default significance level is α = 0.05, which can be changed with the `--alpha` flag:

```bash
python benchmark.py --alpha 0.01  # More strict (fewer ties)
python benchmark.py --alpha 0.10  # Less strict (more ties)
```

- Lower α (e.g., 0.01): Requires stronger evidence to say agents are different → More ties
- Higher α (e.g., 0.10): Requires less evidence to say agents are different → Fewer ties

## Limitations

1. **Independence assumption**: Games are independent, which is true given different seeds
2. **Normality assumption**: t-test assumes normally distributed differences, which may not hold for game scores. However, with 50+ seeds, the Central Limit Theorem provides reasonable approximation
3. **Multiple comparisons**: When comparing many agents, the probability of at least one false positive increases. Consider Bonferroni correction for strict analysis

## Display Format

### Console Output

```
COMPARISON (with 95% confidence intervals)
====================================================================================================
Rank         Agent        Avg Score   Median  Std Dev           95% CI Consistency
----------------------------------------------------------------------------------------------------
🥇 1st       mcts           18472.0    19200    4234  [17638, 19306]      22.9%
🥈 2nd-3rd   expectimax     16710.0    17100    3890  [15954, 17466]      23.3%
...

WIN RATES (% reaching tile threshold)
================================================================================
Agent          2048      1024       512
--------------------------------------------------
mcts           78.0%     94.0%     99.0%
...

GAME LENGTH
============================================================
Agent          Avg Moves      Min      Max
--------------------------------------------------
mcts             892.0      450     1200
...
```

### JSON Output

The `--json` output includes all extended statistics:

```json
{
  "extended": {
    "agent_name": {
      "distribution": {
        "median": 19200.0,
        "std_dev": 4234.5,
        "percentile_25": 16000.0,
        "percentile_75": 21000.0,
        "percentile_90": 23000.0,
        "percentile_99": 24500.0,
        "iqr": 5000.0
      },
      "win_rates": {
        "reached_512": 99.0,
        "reached_1024": 94.0,
        "reached_2048": 78.0
      },
      "game_length": {
        "avg_moves": 892.0,
        "min_moves": 450,
        "max_moves": 1200,
        "median_moves": 900.0
      },
      "consistency": {
        "coefficient_of_variation": 22.9
      }
    }
  }
}
```

### Site Visualization

- **Leaderboard Table**: Shows rank, agent, avg score, median, std dev, 95% CI, and consistency
- **Win Rates Table**: Shows % reaching 2048, 1024, and 512 tiles
- **Win Rates Chart**: Grouped bar chart comparing tile achievement rates
- **Consistency Chart**: Horizontal bar chart showing CV (lower = more consistent)
- **Radar Chart**: Multi-dimensional comparison including score, consistency, 2048 rate, and max tile

## References

- Welch, B. L. (1947). "The generalization of 'Student's' problem when several different population variances are involved"
- scipy.stats.ttest_ind documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
- Coefficient of Variation: https://en.wikipedia.org/wiki/Coefficient_of_variation
