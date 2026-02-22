import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from game2048.agents import BaseAgent, register_agent
from game2048.game import Game2048


@register_agent("td_learning")
class TDLearningAgent(BaseAgent):
    def __init__(
        self,
        weights_file: Optional[str] = None,
        alpha: float = 0.0025,
        gamma: float = 0.99,
        epsilon: float = 0.001,
        seed: Optional[int] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.weights: Dict[Tuple[int, ...], float] = {}
        self.patterns = self._init_patterns()

        if weights_file:
            self.load_weights(weights_file)

    def _init_patterns(self) -> List[List[Tuple[int, int]]]:
        patterns = []

        for row in range(4):
            patterns.append([(row, col) for col in range(4)])

        for col in range(4):
            patterns.append([(row, col) for row in range(4)])

        patterns.append([(0, 0), (0, 1), (1, 0), (1, 1)])
        patterns.append([(0, 2), (0, 3), (1, 2), (1, 3)])
        patterns.append([(2, 0), (2, 1), (3, 0), (3, 1)])
        patterns.append([(2, 2), (2, 3), (3, 2), (3, 3)])

        patterns.append([(0, 0), (0, 1), (0, 2), (1, 2)])
        patterns.append([(0, 3), (0, 2), (0, 1), (1, 1)])
        patterns.append([(3, 0), (3, 1), (3, 2), (2, 2)])
        patterns.append([(3, 3), (3, 2), (3, 1), (2, 1)])

        for i in range(3):
            for j in range(3):
                patterns.append([(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)])

        return patterns

    def _tile_to_feature(self, tile: int) -> int:
        if tile == 0:
            return 0
        return tile.bit_length() - 1

    def _extract_features(self, game: Game2048) -> List[Tuple[Tuple[int, ...], float]]:
        features = []
        for pattern in self.patterns:
            tiles = tuple(self._tile_to_feature(game.grid[r][c]) for r, c in pattern)
            features.append((tiles, 1.0))
        return features

    def _evaluate_features(
        self, features: List[Tuple[Tuple[int, ...], float]]
    ) -> float:
        value = 0.0
        for feature_key, feature_value in features:
            value += self.weights.get(feature_key, 0.0) * feature_value
        return value

    def evaluate(self, game: Game2048) -> float:
        features = self._extract_features(game)
        return self._evaluate_features(features)

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None

        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid)

        best_move = None
        best_value = float("-inf")

        for move in valid:
            new_game = game.clone()
            old_score = new_game.score
            moved = new_game.move(move)
            if not moved:
                continue

            reward = new_game.score - old_score
            value = reward + self.gamma * self.evaluate(new_game)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move if best_move else valid[0]

    def _get_reward(self, game: Game2048, move: str) -> float:
        new_game = game.clone()
        old_score = new_game.score
        new_game.move(move)
        return float(new_game.score - old_score)

    def _td_update(
        self,
        prev_features: List[Tuple[Tuple[int, ...], float]],
        curr_features: List[Tuple[Tuple[int, ...], float]],
        reward: float,
    ):
        v_prev = self._evaluate_features(prev_features)
        v_curr = self._evaluate_features(curr_features)
        delta = reward + self.gamma * v_curr - v_prev

        for feature_key, feature_value in prev_features:
            if feature_key not in self.weights:
                self.weights[feature_key] = 0.0
            self.weights[feature_key] += self.alpha * delta * feature_value

    def _terminal_update(
        self, features: List[Tuple[Tuple[int, ...], float]], final_score: float
    ):
        v_curr = self._evaluate_features(features)
        delta = float(final_score) - v_curr

        for feature_key, feature_value in features:
            if feature_key not in self.weights:
                self.weights[feature_key] = 0.0
            self.weights[feature_key] += self.alpha * delta * feature_value

    def train(
        self,
        num_games: int = 10000,
        verbose: bool = True,
        log_interval: int = 1000,
    ) -> Dict[str, float]:
        scores = []
        max_tiles = []

        for episode in range(1, num_games + 1):
            game = Game2048(seed=None)
            prev_features = None

            while not game.game_over:
                curr_features = self._extract_features(game)

                valid = game.get_valid_moves()
                if not valid:
                    break

                move = self._choose_move_training(game)

                if prev_features is not None:
                    self._td_update(prev_features, curr_features, 0.0)

                prev_features = curr_features
                game.move(move)

            if prev_features is not None:
                self._terminal_update(prev_features, game.score)

            scores.append(game.score)
            max_tiles.append(game.get_max_tile())

            if verbose and episode % log_interval == 0:
                avg_score = sum(scores[-log_interval:]) / log_interval
                max_tile = max(max_tiles[-log_interval:])
                print(
                    f"Episode {episode}/{num_games} - "
                    f"Avg Score (last {log_interval}): {avg_score:.0f} - "
                    f"Max Tile: {max_tile} - "
                    f"Weights: {len(self.weights)}"
                )

        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "best_tile": max(max_tiles),
            "weights_count": len(self.weights),
        }

    def _choose_move_training(self, game: Game2048) -> Optional[str]:
        valid = game.get_valid_moves()
        if not valid:
            return None

        if self.rng.random() < 0.001:
            return self.rng.choice(valid)

        best_move = None
        best_value = float("-inf")

        for move in valid:
            new_game = game.clone()
            old_score = new_game.score
            moved = new_game.move(move)
            if not moved:
                continue

            reward = new_game.score - old_score
            value = reward + self.gamma * self.evaluate(new_game)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move if best_move else valid[0]

    def save_weights(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.weights, f)

    def load_weights(self, filepath: str) -> bool:
        path = Path(filepath)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self.weights = pickle.load(f)
        return True


def train_td_agent(
    num_games: int = 50000,
    weights_file: Optional[str] = None,
    save_interval: int = 5000,
) -> TDLearningAgent:
    agent = TDLearningAgent(weights_file=weights_file)
    results = agent.train(num_games=num_games, verbose=True, log_interval=save_interval)

    print(f"\nTraining completed!")
    print(f"Average Score: {results['avg_score']:.0f}")
    print(f"Max Score: {results['max_score']}")
    print(f"Best Tile: {results['best_tile']}")
    print(f"Total Weights: {results['weights_count']}")

    return agent


if __name__ == "__main__":
    import sys

    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    weights_file = sys.argv[2] if len(sys.argv) > 2 else None

    agent = train_td_agent(num_games=num_games, weights_file=weights_file)
    agent.save_weights("weights/td_learning_weights.pkl")
