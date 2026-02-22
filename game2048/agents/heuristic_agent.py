import math
from typing import Dict, Optional
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("heuristic")
class HeuristicAgent(BaseAgent):
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "empty": 2.7,
            "mono": 1.0,
            "max": 1.0,
            "corner": 2.0,
            "smooth": 1.0,
        }

    def evaluate(self, game: Game2048) -> float:
        score = 0.0
        score += self.weights["empty"] * self._count_empty(game)
        score += self.weights["mono"] * self._monotonicity(game)
        score += self.weights["max"] * self._max_tile_value(game)
        score += self.weights["corner"] * self._corner_bonus(game)
        score += self.weights["smooth"] * self._smoothness(game)
        return score

    def _count_empty(self, game: Game2048) -> float:
        count = sum(1 for row in game.grid for cell in row if cell == 0)
        return float(count)

    def _monotonicity(self, game: Game2048) -> float:
        totals = [0.0, 0.0, 0.0, 0.0]

        for i in range(4):
            current = 0
            next_val = current + 1
            while next_val < 4:
                while next_val < 4 and game.grid[i][next_val] == 0:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                current_value = (
                    math.log2(game.grid[i][current]) if game.grid[i][current] else 0
                )
                next_value = (
                    math.log2(game.grid[i][next_val]) if game.grid[i][next_val] else 0
                )
                if current_value > next_value:
                    totals[0] += next_value - current_value
                elif next_value > current_value:
                    totals[1] += current_value - next_value
                current = next_val
                next_val += 1

        for j in range(4):
            current = 0
            next_val = current + 1
            while next_val < 4:
                while next_val < 4 and game.grid[next_val][j] == 0:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                current_value = (
                    math.log2(game.grid[current][j]) if game.grid[current][j] else 0
                )
                next_value = (
                    math.log2(game.grid[next_val][j]) if game.grid[next_val][j] else 0
                )
                if current_value > next_value:
                    totals[2] += next_value - current_value
                elif next_value > current_value:
                    totals[3] += current_value - next_value
                current = next_val
                next_val += 1

        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def _max_tile_value(self, game: Game2048) -> float:
        max_tile = game.get_max_tile()
        if max_tile <= 0:
            return 0.0
        return math.log2(max_tile)

    def _corner_bonus(self, game: Game2048) -> float:
        max_tile = game.get_max_tile()
        if max_tile <= 0:
            return 0.0
        corners = [
            game.grid[0][0],
            game.grid[0][3],
            game.grid[3][0],
            game.grid[3][3],
        ]
        if max_tile in corners:
            return math.log2(max_tile)
        return 0.0

    def _smoothness(self, game: Game2048) -> float:
        smoothness = 0.0
        for i in range(4):
            for j in range(4):
                if game.grid[i][j] != 0:
                    value = math.log2(game.grid[i][j])
                    if j < 3:
                        right_val = game.grid[i][j + 1]
                        if right_val != 0:
                            smoothness -= abs(value - math.log2(right_val))
                    if i < 3:
                        down_val = game.grid[i + 1][j]
                        if down_val != 0:
                            smoothness -= abs(value - math.log2(down_val))
        return smoothness

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None

        best_move = None
        best_score = float("-inf")

        for move in valid:
            new_game = game.clone()
            new_game.move(move)
            score = self.evaluate(new_game)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
