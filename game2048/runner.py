from typing import Callable, Optional
from game2048.game import Game2048


class GameRunner:
    def __init__(
        self,
        game: Game2048,
        choose_move: Callable[[Game2048], Optional[str]],
        verbose: bool = False,
    ):
        self.game = game
        self.choose_move = choose_move
        self.verbose = verbose
        self.moves = 0

    def run(self) -> int:
        if self.verbose:
            print("Initial state:")
            print(self.game)

        while not self.game.game_over:
            move = self.choose_move(self.game)
            if move is None:
                break
            self.game.move(move)
            self.moves += 1
            if self.verbose:
                print(f"Move {self.moves}: {move}")
                print(self.game)

        return self.game.score

    def get_results(self) -> dict:
        return {
            "score": self.game.score,
            "max_tile": self.game.get_max_tile(),
            "moves": self.moves,
            "game_over": self.game.game_over,
        }
