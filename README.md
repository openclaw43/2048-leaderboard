# 2048 Leaderboard

A complete 2048 game implementation with multiple AI agents, benchmarking, and leaderboard support.

## Features

- **Complete 2048 Implementation**: Faithful to the original game mechanics
- **Multiple AI Agents**: Random, heuristic, and search-based agents
- **Benchmarking**: Automated testing across multiple seeds
- **Reproducibility**: All games use seed-based randomization
- **GitHub Actions**: Automated benchmarking on every PR
- **OpenCode Integration**: Custom slash commands for development

## Quick Start

Using [Astral UV](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/openclaw43/2048-leaderboard.git
cd 2048-leaderboard

# Install dependencies
uv sync

# Run the game with an agent
uv run python main.py --agent random --seed 42

# Run with verbose output
uv run python main.py --agent random --seed 42 --verbose

# Run tests
uv run pytest test_game.py -v

# Run benchmarks
uv run python benchmark.py

# Run benchmarks with JSON output
uv run python benchmark.py --json --output results.json
```

## Available Agents

| Agent | Description | Avg Score |
|-------|-------------|-----------|
| `random` | Makes random valid moves | ~1200 |
| `rightleft` | Alternates right/left | ~500 |
| `rightdown` | Alternates right/down | ~1100 |

## Project Structure

```
2048-leaderboard/
├── game2048/              # Main package
│   ├── __init__.py
│   ├── game.py           # Core 2048 game logic
│   ├── runner.py         # Game runner for agents
│   └── agents/           # Agent implementations
│       ├── __init__.py
│       ├── random_agent.py
│       ├── rightleft_agent.py
│       └── rightdown_agent.py
├── .github/
│   └── workflows/
│       └── benchmark.yml # CI/CD for benchmarks
├── .opencode/
│   └── commands/         # Custom OpenCode commands
│       └── implement-issue.md
├── scripts/
│   └── implement_issue.py
├── main.py               # CLI entry point
├── benchmark.py          # Benchmarking script
├── test_game.py          # Unit tests
├── pyproject.toml        # UV project config
└── README.md
```

## OpenCode Custom Commands

This repository includes custom slash commands for OpenCode:

### `/implement-issue <issue-number>`

Automatically implements a GitHub issue:

```bash
opencode /implement-issue 2
```

This will:
1. Read the issue details
2. Create a feature branch
3. Implement the feature
4. Run tests and benchmarks
5. Open a pull request

See [.opencode/README.md](.opencode/README.md) for more details.

## Python API

```python
from game2048 import Game2048, GameRunner
from game2048.agents import RandomAgent, list_agents

# List available agents
print(list_agents())  # ['random', 'rightleft', 'rightdown']

# Create a game with seed
game = Game2048(seed=42)
agent = RandomAgent(seed=42)

# Run the game
runner = GameRunner(game, agent.choose_move, verbose=True)
score = runner.run()

# Get detailed results
results = runner.get_results()
print(f"Score: {results['score']}")
print(f"Max Tile: {results['max_tile']}")
print(f"Moves: {results['moves']}")
```

## Development

### Adding a New Agent

1. Create a new file in `game2048/agents/`
2. Implement the agent class inheriting from `BaseAgent`
3. Register with `@register_agent("name")`
4. Add tests in `test_game.py`
5. Run benchmarks

Example:

```python
from game2048.agents import register_agent, BaseAgent
from game2048.game import Game2048
from typing import Optional

@register_agent("my_agent")
class MyAgent(BaseAgent):
    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None
        # Your logic here
        return valid[0]
```

### Running Benchmarks

```bash
# Run all agents on seeds 1-50
uv run python benchmark.py

# Run with specific number of seeds
uv run python benchmark.py --seeds 100

# Output as JSON
uv run python benchmark.py --json --output results.json

# Verbose per-seed output
uv run python benchmark.py --verbose
```

## Contributing

1. Check open issues for algorithm ideas: https://github.com/openclaw43/2048-leaderboard/issues
2. Create a feature branch: `git checkout -b feature/my-agent`
3. Implement and test
4. Open a PR (all changes go through PR review)

## Algorithm Ideas

See [GitHub Issues](https://github.com/openclaw43/2048-leaderboard/issues) for algorithm implementations:

- Corner Strategy (#2)
- Snake/Monotonic (#3)
- Greedy (#4)
- Heuristic Evaluation (#5)
- Expectimax (#6)
- Monte Carlo Tree Search (#7)
- TD-Learning (#8)

## License

MIT
