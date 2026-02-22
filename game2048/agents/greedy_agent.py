from typing import List, Optional, Tuple
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("greedy")
class GreedyAgent(BaseAgent):
    def __init__(self, tie_breaker: List[str] = None):
        self.tie_breaker = tie_breaker or ["down", "right", "left", "up"]

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None

        best_move = None
        best_score = -1

        for move in valid:
            score_gain = self._simulate_score_gain(game, move)
            if score_gain > best_score:
                best_score = score_gain
                best_move = move
            elif score_gain == best_score and best_move is not None:
                for preferred in self.tie_breaker:
                    if preferred == move:
                        best_move = move
                        break
                    if preferred == best_move:
                        break

        return best_move

    def _simulate_score_gain(self, game: Game2048, move: str) -> int:
        grid = game.grid
        if move == Game2048.LEFT:
            _, score, _ = self._move_left(grid)
        elif move == Game2048.RIGHT:
            reversed_grid = self._reverse_rows(grid)
            _, score, _ = self._move_left(reversed_grid)
        elif move == Game2048.UP:
            transposed = self._transpose(grid)
            _, score, _ = self._move_left(transposed)
        elif move == Game2048.DOWN:
            transposed = self._transpose(grid)
            reversed_grid = self._reverse_rows(transposed)
            _, score, _ = self._move_left(reversed_grid)
        else:
            score = 0
        return score

    def _compress_row(self, row: List[int]) -> Tuple[List[int], int]:
        compressed = [x for x in row if x != 0]
        score = 0
        result = []
        i = 0
        while i < len(compressed):
            if i + 1 < len(compressed) and compressed[i] == compressed[i + 1]:
                merged_value = compressed[i] * 2
                result.append(merged_value)
                score += merged_value
                i += 2
            else:
                result.append(compressed[i])
                i += 1
        while len(result) < 4:
            result.append(0)
        return result, score

    def _move_left(self, grid: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
        new_grid = []
        total_score = 0
        moved = False
        for row in grid:
            new_row, score = self._compress_row(row)
            new_grid.append(new_row)
            total_score += score
            if new_row != row:
                moved = True
        return new_grid, total_score, moved

    def _transpose(self, grid: List[List[int]]) -> List[List[int]]:
        return [[grid[j][i] for j in range(4)] for i in range(4)]

    def _reverse_rows(self, grid: List[List[int]]) -> List[List[int]]:
        return [row[::-1] for row in grid]
