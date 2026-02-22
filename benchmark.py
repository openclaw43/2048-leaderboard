#!/usr/bin/env python3
"""Test 2048 game across multiple seeds and report statistics."""

from game2048.game import Game2048
from game2048.agents import RandomAgent


def run_game(seed: int) -> dict:
    """Run a single game with given seed and return results."""
    game = Game2048(seed=seed)
    agent = RandomAgent(seed=seed)

    moves = 0
    while not game.game_over:
        move = agent.choose_move(game)
        if move is None:
            break
        game.move(move)
        moves += 1

    return {
        "seed": seed,
        "score": game.score,
        "max_tile": game.get_max_tile(),
        "moves": moves,
    }


def main():
    results = []

    print("Testing 2048 game with seeds 1-50...")
    print("=" * 60)

    for seed in range(1, 51):
        result = run_game(seed)
        results.append(result)
        print(
            f"Seed {seed:2d}: Score={result['score']:5d}, Max={result['max_tile']:4d}, Moves={result['moves']:3d}"
        )

    print("=" * 60)

    scores = [r["score"] for r in results]
    max_tiles = [r["max_tile"] for r in results]
    moves = [r["moves"] for r in results]

    print("\n📊 STATISTICS")
    print("-" * 40)
    print(
        f"Score:     Avg={sum(scores) / len(scores):.1f}, Min={min(scores)}, Max={max(scores)}"
    )
    print(
        f"Max Tile:  Avg={sum(max_tiles) / len(max_tiles):.1f}, Min={min(max_tiles)}, Max={max(max_tiles)}"
    )
    print(
        f"Moves:     Avg={sum(moves) / len(moves):.1f}, Min={min(moves)}, Max={max(moves)}"
    )

    print("\n🎯 MAX TILE DISTRIBUTION")
    print("-" * 40)
    from collections import Counter

    tile_counts = Counter(max_tiles)
    for tile in sorted(tile_counts.keys()):
        count = tile_counts[tile]
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"{tile:4d}: {count:2d} games ({pct:5.1f}%) {bar}")

    print("\n🏆 TOP 5 SCORES")
    print("-" * 40)
    top5 = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(
            f"{i}. Seed {r['seed']:2d}: Score={r['score']}, Max={r['max_tile']}, Moves={r['moves']}"
        )

    print("\n💩 BOTTOM 5 SCORES")
    print("-" * 40)
    bottom5 = sorted(results, key=lambda x: x["score"])[:5]
    for i, r in enumerate(bottom5, 1):
        print(
            f"{i}. Seed {r['seed']:2d}: Score={r['score']}, Max={r['max_tile']}, Moves={r['moves']}"
        )


if __name__ == "__main__":
    main()
