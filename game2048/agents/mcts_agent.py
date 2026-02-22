import random
from typing import Optional
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("mcts")
class MCTSAgent(BaseAgent):
    def __init__(self, simulations: int = 100, exploration_constant: float = 1.414):
        self.simulations = simulations
        self.rollouts_per_move = max(1, simulations // 4)
        self.max_depth = 4

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]

        best_move = valid[0]
        best_score = -1.0

        for move in valid:
            total_score = 0.0
            for _ in range(self.rollouts_per_move):
                sim_game = game.clone()
                sim_game.move(move)
                score = self._rollout(sim_game)
                total_score += score
            avg_score = total_score / self.rollouts_per_move

            if avg_score > best_score:
                best_score = avg_score
                best_move = move

        return best_move

    def _rollout(self, game: Game2048) -> float:
        initial_score = game.score
        max_tile = game.get_max_tile()
        depth = 0

        while not game.game_over and depth < self.max_depth:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            game.move(move)
            depth += 1

        score_gain = game.score - initial_score
        empty_cells = sum(1 for row in game.grid for cell in row if cell == 0)
        final_max_tile = game.get_max_tile()
        tile_improvement = final_max_tile - max_tile

        return float(score_gain + empty_cells * 100 + tile_improvement * 10)
