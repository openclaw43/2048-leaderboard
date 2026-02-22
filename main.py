#!/usr/bin/env python3
import argparse
from game2048.game import Game2048
from game2048.agents import get_agent, list_agents


def main():
    parser = argparse.ArgumentParser(description="Play 2048 with an agent")
    parser.add_argument(
        "--agent",
        choices=list_agents(),
        default="random",
        help=f"Agent type to use (default: random)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print game state after each move"
    )
    args = parser.parse_args()

    game = Game2048(seed=args.seed)
    agent_factory = get_agent(args.agent)

    if agent_factory is None:
        print(f"Unknown agent: {args.agent}")
        return

    if args.agent == "random":
        agent = agent_factory(seed=args.seed)
    else:
        agent = agent_factory()

    from game2048.runner import GameRunner

    runner = GameRunner(game, agent.choose_move, verbose=args.verbose)
    runner.run()

    results = runner.get_results()
    print(f"\nGame Over!")
    print(f"Final Score: {results['score']}")
    print(f"Max Tile: {results['max_tile']}")
    print(f"Total Moves: {results['moves']}")


if __name__ == "__main__":
    main()
