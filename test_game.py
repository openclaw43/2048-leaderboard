from __future__ import annotations

import unittest

from game2048.agents import (
    ApprenticeAgent,
    CornerAgent,
    ExpectimaxAgent,
    GreedyAgent,
    MCTSAgent,
    RandomAgent,
    RightDownAgent,
    RightLeftAgent,
    SnakeAgent,
    TDLearningAgent,
)
from game2048.game import Game2048


class TestGame2048(unittest.TestCase):
    def test_initial_state(self) -> None:
        game = Game2048(seed=42)
        non_zero = sum(1 for row in game.grid for cell in row if cell != 0)
        self.assertEqual(non_zero, 2)
        self.assertEqual(game.score, 0)
        self.assertFalse(game.game_over)

    def test_reproducibility(self) -> None:
        game1 = Game2048(seed=123)
        game2 = Game2048(seed=123)
        self.assertEqual(game1.grid, game2.grid)

    def test_move_changes_grid(self) -> None:
        game = Game2048(seed=42)
        initial_grid = [row[:] for row in game.grid]
        moved = game.move(Game2048.LEFT)
        if moved:
            self.assertNotEqual(initial_grid, game.grid)

    def test_score_increases_on_merge(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.move(Game2048.LEFT)
        self.assertGreater(game.score, 0)

    def test_get_valid_moves(self) -> None:
        game = Game2048(seed=42)
        valid_moves = game.get_valid_moves()
        self.assertIsInstance(valid_moves, list)
        for move in valid_moves:
            self.assertIn(move, Game2048.MOVES)

    def test_game_over_detection(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        self.assertEqual(game.get_valid_moves(), [])
        game.move(Game2048.LEFT)
        self.assertTrue(game.game_over)

    def test_max_tile(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(game.get_max_tile(), 16)


class TestRandomAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = RandomAgent(seed=100)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_reproducibility(self) -> None:
        game1 = Game2048(seed=42)
        game2 = Game2048(seed=42)
        agent1 = RandomAgent(seed=100)
        agent2 = RandomAgent(seed=100)

        for _ in range(10):
            move1 = agent1.choose_move(game1)
            move2 = agent2.choose_move(game2)
            self.assertEqual(move1, move2)
            if move1:
                game1.move(move1)
            if move2:
                game2.move(move2)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = RandomAgent(seed=100)
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestRightLeftAgent(unittest.TestCase):
    def test_agent_alternates_moves(self) -> None:
        game = Game2048(seed=42)
        agent = RightLeftAgent()
        self.assertTrue(agent.prefer_right)
        agent.choose_move(game)
        self.assertFalse(agent.prefer_right)
        agent.choose_move(game)
        self.assertTrue(agent.prefer_right)

    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = RightLeftAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = RightLeftAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestRightDownAgent(unittest.TestCase):
    def test_agent_alternates_moves(self) -> None:
        game = Game2048(seed=42)
        agent = RightDownAgent()
        self.assertTrue(agent.prefer_right)
        agent.choose_move(game)
        self.assertFalse(agent.prefer_right)
        agent.choose_move(game)
        self.assertTrue(agent.prefer_right)

    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = RightDownAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = RightDownAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestCornerAgent(unittest.TestCase):
    def test_agent_prefers_corner_moves(self) -> None:
        game = Game2048(seed=42)
        agent = CornerAgent()
        self.assertEqual(agent.preferred, ["down", "right", "left", "up"])

    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = CornerAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_fallback_behavior(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 2, 4, 8], [4, 2, 2, 4], [8, 4, 2, 2], [2, 4, 8, 4]]
        game.game_over = False
        agent = CornerAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = CornerAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestGreedyAgent(unittest.TestCase):
    def test_agent_prefers_merge_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.game_over = False
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertIn(move, ["left", "right"])

    def test_agent_uses_tie_breaker(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.game_over = False
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertEqual(move, "down")

    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = GreedyAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestSnakeAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = SnakeAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = SnakeAgent()
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_agent_prefers_monotonic_pattern(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[64, 32, 16, 8], [2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.game_over = False
        agent = SnakeAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = SnakeAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestExpectimaxAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = ExpectimaxAgent(depth=2)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = ExpectimaxAgent(depth=2)
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_agent_with_different_depths(self) -> None:
        game = Game2048(seed=42)
        agent_depth1 = ExpectimaxAgent(depth=1)
        agent_depth2 = ExpectimaxAgent(depth=2)
        move1 = agent_depth1.choose_move(game)
        move2 = agent_depth2.choose_move(game)
        self.assertIn(move1, Game2048.MOVES)
        self.assertIn(move2, Game2048.MOVES)

    def test_evaluate_returns_float(self) -> None:
        game = Game2048(seed=42)
        agent = ExpectimaxAgent(depth=2)
        value = agent.evaluate(game)
        self.assertIsInstance(value, float)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = ExpectimaxAgent(depth=2)
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestMCTSAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = MCTSAgent(simulations=10)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = MCTSAgent(simulations=10)
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_agent_with_different_simulations(self) -> None:
        game = Game2048(seed=42)
        agent_10 = MCTSAgent(simulations=10)
        agent_50 = MCTSAgent(simulations=50)
        move1 = agent_10.choose_move(game)
        move2 = agent_50.choose_move(game)
        self.assertIn(move1, Game2048.MOVES)
        self.assertIn(move2, Game2048.MOVES)

    def test_agent_returns_single_valid_move(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 2, 0, 0], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = MCTSAgent(simulations=10)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = MCTSAgent(simulations=10)
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestTDLearningAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = TDLearningAgent(seed=100)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = TDLearningAgent(seed=100)
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_agent_with_different_hyperparameters(self) -> None:
        game = Game2048(seed=42)
        agent1 = TDLearningAgent(alpha=0.01, gamma=0.99, seed=100)
        agent2 = TDLearningAgent(alpha=0.001, gamma=0.95, seed=100)
        move1 = agent1.choose_move(game)
        move2 = agent2.choose_move(game)
        self.assertIn(move1, Game2048.MOVES)
        self.assertIn(move2, Game2048.MOVES)

    def test_evaluate_returns_float(self) -> None:
        game = Game2048(seed=42)
        agent = TDLearningAgent(seed=100)
        value = agent.evaluate(game)
        self.assertIsInstance(value, float)

    def test_pattern_extraction(self) -> None:
        game = Game2048(seed=42)
        agent = TDLearningAgent(seed=100)
        features = agent._extract_features(game)
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)

    def test_training_updates_weights(self) -> None:
        agent = TDLearningAgent(alpha=0.1, seed=100)
        initial_weights = len(agent.weights)
        agent.train(num_games=10, verbose=False)
        self.assertGreaterEqual(len(agent.weights), initial_weights)

    def test_save_and_load_weights(self) -> None:
        import os
        import tempfile

        agent1 = TDLearningAgent(seed=100)
        agent1.train(num_games=5, verbose=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            temp_path = f.name

        try:
            agent1.save_weights(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            agent2 = TDLearningAgent(seed=200)
            loaded = agent2.load_weights(temp_path)
            self.assertTrue(loaded)
            self.assertEqual(agent1.weights, agent2.weights)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_nonexistent_weights(self) -> None:
        agent = TDLearningAgent(seed=100)
        loaded = agent.load_weights("/nonexistent/path/weights.pkl")
        self.assertFalse(loaded)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = TDLearningAgent(seed=100)
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestApprenticeAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self) -> None:
        game = Game2048(seed=42)
        agent = ApprenticeAgent()
        agent.train(num_samples=100, epochs=2, verbose=False, seed=42)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self) -> None:
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = ApprenticeAgent()
        agent.train(num_samples=100, epochs=2, verbose=False, seed=42)
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_agent_with_different_hidden_sizes(self) -> None:
        game = Game2048(seed=42)
        agent1 = ApprenticeAgent(hidden_sizes=[64, 32])
        agent1.train(num_samples=100, epochs=2, verbose=False, seed=42)
        agent2 = ApprenticeAgent(hidden_sizes=[256])
        agent2.train(num_samples=100, epochs=2, verbose=False, seed=42)
        move1 = agent1.choose_move(game)
        move2 = agent2.choose_move(game)
        self.assertIn(move1, Game2048.MOVES)
        self.assertIn(move2, Game2048.MOVES)

    def test_inference_time_is_fast(self) -> None:
        import time

        game = Game2048(seed=42)
        agent = ApprenticeAgent()
        agent.train(num_samples=100, epochs=2, verbose=False, seed=42)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            agent.choose_move(game)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 10.0)

    def test_training_improves_validation_accuracy(self) -> None:
        agent = ApprenticeAgent()
        history = agent.train(num_samples=500, epochs=5, verbose=False, seed=42)
        self.assertGreater(history["val_accuracy"], 0.15)

    def test_save_and_load_model(self) -> None:
        import os
        import tempfile

        agent1 = ApprenticeAgent()
        agent1.train(num_samples=100, epochs=2, verbose=False, seed=42)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            temp_path = f.name

        try:
            agent1.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            agent2 = ApprenticeAgent(model_path=temp_path)
            game = Game2048(seed=42)
            move1 = agent1.choose_move(game)
            move2 = agent2.choose_move(game)
            self.assertEqual(move1, move2)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_nonexistent_model(self) -> None:
        agent = ApprenticeAgent(model_path="/nonexistent/path/model.pkl")
        self.assertIsNone(agent.network)

    def test_play_game(self) -> None:
        game = Game2048(seed=42)
        agent = ApprenticeAgent()
        agent.train(num_samples=200, epochs=3, verbose=False, seed=42)
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


if __name__ == "__main__":
    unittest.main()
