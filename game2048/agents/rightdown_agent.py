from typing import Optional
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("rightdown")
class RightDownAgent(BaseAgent):
    def __init__(self):
        self.prefer_right = True

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        first = Game2048.RIGHT if self.prefer_right else Game2048.DOWN
        second = Game2048.DOWN if self.prefer_right else Game2048.RIGHT
        self.prefer_right = not self.prefer_right
        if first in valid_moves:
            return first
        if second in valid_moves:
            return second
        return valid_moves[0] if valid_moves else None
