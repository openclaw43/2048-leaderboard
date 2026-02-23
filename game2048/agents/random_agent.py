from __future__ import annotations

import random
from typing import Optional

from game2048.agents import BaseAgent, register_agent
from game2048.game import Game2048


@register_agent("random")
class RandomAgent(BaseAgent):
    rng: random.Random

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return self.rng.choice(valid_moves)
