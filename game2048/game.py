import random
from typing import List, Optional, Tuple


class Game2048:
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    MOVES = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.game_over = False
        self._add_random_tile()
        self._add_random_tile()

    def _add_random_tile(self) -> bool:
        empty_cells = [
            (i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0
        ]
        if not empty_cells:
            return False
        i, j = self.rng.choice(empty_cells)
        self.grid[i][j] = 4 if self.rng.random() < 0.1 else 2
        return True

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

    def _move_left(self) -> Tuple[List[List[int]], int, bool]:
        new_grid = []
        total_score = 0
        moved = False
        for row in self.grid:
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

    def move(self, direction: str) -> bool:
        if self.game_over or direction not in self.MOVES:
            return False

        if direction == self.LEFT:
            new_grid, score, moved = self._move_left()
        elif direction == self.RIGHT:
            self.grid = self._reverse_rows(self.grid)
            new_grid, score, moved = self._move_left()
            new_grid = self._reverse_rows(new_grid)
            self.grid = self._reverse_rows(self.grid)
        elif direction == self.UP:
            self.grid = self._transpose(self.grid)
            new_grid, score, moved = self._move_left()
            new_grid = self._transpose(new_grid)
            self.grid = self._transpose(self.grid)
        elif direction == self.DOWN:
            self.grid = self._transpose(self.grid)
            self.grid = self._reverse_rows(self.grid)
            new_grid, score, moved = self._move_left()
            new_grid = self._reverse_rows(new_grid)
            new_grid = self._transpose(new_grid)
            self.grid = self._reverse_rows(self.grid)
            self.grid = self._transpose(self.grid)

        if moved:
            self.grid = new_grid
            self.score += score
            self._add_random_tile()
        if not self._can_move():
            self.game_over = True
        return moved

    def _can_move(self) -> bool:
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 0:
                    return True
                if j < 3 and self.grid[i][j] == self.grid[i][j + 1]:
                    return True
                if i < 3 and self.grid[i][j] == self.grid[i + 1][j]:
                    return True
        return False

    def get_valid_moves(self) -> List[str]:
        return [m for m in self.MOVES if self._would_move(m)]

    def _would_move(self, direction: str) -> bool:
        old_grid = [row[:] for row in self.grid]
        old_score = self.score
        old_game_over = self.game_over

        if direction == self.LEFT:
            new_grid, _, moved = self._move_left()
        elif direction == self.RIGHT:
            self.grid = self._reverse_rows(self.grid)
            new_grid, _, moved = self._move_left()
            new_grid = self._reverse_rows(new_grid)
            self.grid = self._reverse_rows(self.grid)
        elif direction == self.UP:
            self.grid = self._transpose(self.grid)
            new_grid, _, moved = self._move_left()
            new_grid = self._transpose(new_grid)
            self.grid = self._transpose(self.grid)
        elif direction == self.DOWN:
            self.grid = self._transpose(self.grid)
            self.grid = self._reverse_rows(self.grid)
            new_grid, _, moved = self._move_left()
            new_grid = self._reverse_rows(new_grid)
            new_grid = self._transpose(new_grid)
            self.grid = self._reverse_rows(self.grid)
            self.grid = self._transpose(self.grid)

        self.grid = old_grid
        self.score = old_score
        self.game_over = old_game_over
        return moved

    def get_max_tile(self) -> int:
        return max(max(row) for row in self.grid)

    def clone(self) -> "Game2048":
        new_game = Game2048.__new__(Game2048)
        new_game.rng = random.Random()
        new_game.rng.setstate(self.rng.getstate())
        new_game.grid = [row[:] for row in self.grid]
        new_game.score = self.score
        new_game.game_over = self.game_over
        return new_game

    def __str__(self) -> str:
        result = f"Score: {self.score}\n"
        for row in self.grid:
            result += " ".join(f"{x:5}" for x in row) + "\n"
        return result
