# 2048 Leaderboard

A complete 2048 game implementation with a random agent and seed support for reproducibility.

## Setup

Using [Astral UV](https://docs.astral.sh/uv/):

```bash
# Install dependencies (if any)
uv sync

# Run the game
uv run python main.py --seed 42

# Run with verbose output
uv run python main.py --seed 42 --verbose

# Run tests
uv run pytest test_game.py -v
```

## Files

- **game.py** - Core 2048 game logic with seed support
- **agent.py** - RandomAgent implementation
- **main.py** - CLI entry point
- **test_game.py** - Unit tests

## Usage

```python
from game import Game2048
from agent import RandomAgent

# Create a reproducible game
game = Game2048(seed=42)
agent = RandomAgent(seed=42)

# Play until game over
while not game.game_over:
    move = agent.choose_move(game)
    if move:
        game.move(move)

print(f"Final score: {game.score}")
```

## License

MIT
