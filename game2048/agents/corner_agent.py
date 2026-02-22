from typing import Optional
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("corner")
class CornerAgent(BaseAgent):
    def __init__(self, corner: str = "bottom-right"):
        self.corner = corner
        self.preferred = ["down", "right", "left", "up"]

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None
        for move in self.preferred:
            if move in valid:
                return move
        return valid[0]
