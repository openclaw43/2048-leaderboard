from __future__ import annotations

from typing import Callable, Dict, Optional, Type, Union

from game2048.game import Game2048


AgentFactory = Callable[..., "BaseAgent"]


_registry: Dict[str, AgentFactory] = {}


def register_agent(name: str) -> Callable[[Type[BaseAgent]], Type[BaseAgent]]:
    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        _registry[name] = cls
        return cls

    return decorator


def get_agent(name: str) -> Optional[AgentFactory]:
    return _registry.get(name)


def list_agents() -> list[str]:
    return list(_registry.keys())


class BaseAgent:
    def choose_move(self, game: Game2048) -> Optional[str]:
        raise NotImplementedError

    def play_game(self, game: Game2048) -> int:
        from game2048.runner import GameRunner

        runner = GameRunner(game, self.choose_move)
        return runner.run()


from game2048.agents.random_agent import RandomAgent
from game2048.agents.rightleft_agent import RightLeftAgent
from game2048.agents.rightdown_agent import RightDownAgent
from game2048.agents.corner_agent import CornerAgent
from game2048.agents.greedy_agent import GreedyAgent
from game2048.agents.snake_agent import SnakeAgent
from game2048.agents.expectimax_agent import ExpectimaxAgent
from game2048.agents.mcts_agent import MCTSAgent
from game2048.agents.td_learning_agent import TDLearningAgent
from game2048.agents.apprentice_agent import ApprenticeAgent

__all__: list[str] = [
    "BaseAgent",
    "RandomAgent",
    "RightLeftAgent",
    "RightDownAgent",
    "CornerAgent",
    "GreedyAgent",
    "SnakeAgent",
    "ExpectimaxAgent",
    "MCTSAgent",
    "TDLearningAgent",
    "ApprenticeAgent",
    "register_agent",
    "get_agent",
    "list_agents",
]
