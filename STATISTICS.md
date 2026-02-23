# Statistical Methodology

This document describes the statistical methods used in the 2048 AI Leaderboard to provide confidence intervals and rank ranges.

## Overview

Instead of showing exact rankings that may be misleading due to random variance, we use statistical significance testing to group agents into performance tiers.

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

### PR Comments

| Rank | Agent | Avg Score | 95% CI | Max Score | Avg Max Tile |
|------|-------|-----------|--------|-----------|-------------|
| 1st | mcts | 18,472 | [17,638, 19,306] | 24,580 | 2,048 |
| 2nd-3rd | expectimax | 16,710 | [15,954, 17,466] | 22,140 | 1,932 |
| 2nd-3rd | heuristic | 16,500 | [15,711, 17,289] | 21,890 | 1,876 |

### Site Visualization

- Error bars on bar charts show 95% confidence intervals
- Agents with overlapping confidence intervals may be tied
- Hover tooltips show exact confidence interval values

## References

- Welch, B. L. (1947). "The generalization of 'Student's' problem when several different population variances are involved"
- scipy.stats.ttest_ind documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
