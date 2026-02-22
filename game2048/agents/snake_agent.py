import math
from typing import Optional
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("snake")
class SnakeAgent(BaseAgent):
    SNAKE_WEIGHTS = [
        [15, 14, 13, 12],
        [8, 9, 10, 11],
        [7, 6, 5, 4],
        [0, 1, 2, 3],
    ]

    def __init__(self):
        pass

    def evaluate(self, game: Game2048) -> float:
        score = 0.0
        score += self._snake_pattern_score(game)
        score += self._monotonicity(game) * 1.0
        score += self._smoothness(game) * 0.5
        score += self._empty_cells(game) * 2.0
        score += self._merge_potential(game) * 1.5
        return score

    def _snake_pattern_score(self, game: Game2048) -> float:
        score = 0.0
        for i in range(4):
            for j in range(4):
                if game.grid[i][j] != 0:
                    weight = self.SNAKE_WEIGHTS[i][j]
                    score += game.grid[i][j] * weight
        return math.log2(max(score, 1))

    def _monotonicity(self, game: Game2048) -> float:
        scores = [0.0, 0.0, 0.0, 0.0]

        for i in range(4):
            for j in range(3):
                current = game.grid[i][j]
                next_val = game.grid[i][j + 1]
                if current > next_val:
                    scores[0] += math.log2(current) - math.log2(max(next_val, 1))
                elif next_val > current:
                    scores[1] += math.log2(next_val) - math.log2(max(current, 1))

        for j in range(4):
            for i in range(3):
                current = game.grid[i][j]
                next_val = game.grid[i + 1][j]
                if current > next_val:
                    scores[2] += math.log2(current) - math.log2(max(next_val, 1))
                elif next_val > current:
                    scores[3] += math.log2(next_val) - math.log2(max(current, 1))

        return max(scores[0], scores[1]) + max(scores[2], scores[3])

    def _smoothness(self, game: Game2048) -> float:
        smoothness = 0.0
        for i in range(4):
            for j in range(4):
                if game.grid[i][j] != 0:
                    value = math.log2(game.grid[i][j])
                    if j < 3 and game.grid[i][j + 1] != 0:
                        smoothness -= abs(value - math.log2(game.grid[i][j + 1]))
                    if i < 3 and game.grid[i + 1][j] != 0:
                        smoothness -= abs(value - math.log2(game.grid[i + 1][j]))
        return smoothness

    def _empty_cells(self, game: Game2048) -> float:
        count = sum(1 for row in game.grid for cell in row if cell == 0)
        return float(count)

    def _merge_potential(self, game: Game2048) -> float:
        merges = 0
        for i in range(4):
            for j in range(4):
                if game.grid[i][j] != 0:
                    if j < 3 and game.grid[i][j] == game.grid[i][j + 1]:
                        merges += game.grid[i][j]
                    if i < 3 and game.grid[i][j] == game.grid[i + 1][j]:
                        merges += game.grid[i][j]
        return math.log2(max(merges, 1))

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
