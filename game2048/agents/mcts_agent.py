import math
import random
from typing import List, Optional
from game2048.game import Game2048
from game2048.agents import register_agent, BaseAgent


@register_agent("mcts")
class MCTSAgent(BaseAgent):
    def __init__(
        self,
        simulations: int = 100,
        exploration_constant: float = 1.414,
        max_rollout_depth: int = 20,
    ):
        self.simulations = simulations
        self.C = exploration_constant
        self.max_rollout_depth = max_rollout_depth

    class Node:
        def __init__(
            self, game_state: Game2048, parent=None, move: Optional[str] = None
        ):
            self.game_state = game_state
            self.parent = parent
            self.move = move
            self.children: List["MCTSAgent.Node"] = []
            self.visits = 0
            self.total_score = 0
            self.untried_moves = game_state.get_valid_moves()

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]

        root = self.Node(game.clone())

        for _ in range(self.simulations):
            node = self._select(root)
            if node.untried_moves:
                node = self._expand(node)
            score = self._simulate(node.game_state)
            self._backpropagate(node, score)

        if not root.children:
            return valid[0]

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _select(self, node: Node) -> Node:
        while not node.untried_moves and node.children:
            node = self._ucb_select(node)
        return node

    def _expand(self, node: Node) -> Node:
        move = node.untried_moves.pop()
        new_game = node.game_state.clone()
        new_game.move(move)

        child = self.Node(new_game, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, game: Game2048) -> float:
        sim_game = game.clone()
        initial_score = sim_game.score
        depth = 0

        while not sim_game.game_over and depth < self.max_rollout_depth:
            valid_moves = sim_game.get_valid_moves()
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            sim_game.move(move)
            depth += 1

        score_gain = sim_game.score - initial_score
        empty_cells = sum(1 for row in sim_game.grid for cell in row if cell == 0)
        max_tile = sim_game.get_max_tile()

        return score_gain + empty_cells * 10 + max_tile

    def _backpropagate(self, node: Node, score: float):
        current = node
        while current is not None:
            current.visits += 1
            current.total_score += score
            current = current.parent

    def _ucb_select(self, node: Node) -> Node:
        log_parent = math.log(node.visits)
        return max(node.children, key=lambda c: self._ucb_value(c, log_parent))

    def _ucb_value(self, child: Node, log_parent: float) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.total_score / child.visits
        exploration = self.C * math.sqrt(log_parent / child.visits)
        return exploitation + exploration
