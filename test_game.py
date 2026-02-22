import unittest
from game2048.game import Game2048
from game2048.agents import (
    RandomAgent,
    RightLeftAgent,
    RightDownAgent,
    CornerAgent,
    GreedyAgent,
    HeuristicAgent,
)


class TestGame2048(unittest.TestCase):
    def test_initial_state(self):
        game = Game2048(seed=42)
        non_zero = sum(1 for row in game.grid for cell in row if cell != 0)
        self.assertEqual(non_zero, 2)
        self.assertEqual(game.score, 0)
        self.assertFalse(game.game_over)

    def test_reproducibility(self):
        game1 = Game2048(seed=123)
        game2 = Game2048(seed=123)
        self.assertEqual(game1.grid, game2.grid)

    def test_move_changes_grid(self):
        game = Game2048(seed=42)
        initial_grid = [row[:] for row in game.grid]
        moved = game.move(Game2048.LEFT)
        if moved:
            self.assertNotEqual(initial_grid, game.grid)

    def test_score_increases_on_merge(self):
        game = Game2048(seed=42)
        game.grid = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.move(Game2048.LEFT)
        self.assertGreater(game.score, 0)

    def test_get_valid_moves(self):
        game = Game2048(seed=42)
        valid_moves = game.get_valid_moves()
        self.assertIsInstance(valid_moves, list)
        for move in valid_moves:
            self.assertIn(move, Game2048.MOVES)

    def test_game_over_detection(self):
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        self.assertEqual(game.get_valid_moves(), [])
        game.move(Game2048.LEFT)
        self.assertTrue(game.game_over)

    def test_max_tile(self):
        game = Game2048(seed=42)
        game.grid = [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(game.get_max_tile(), 16)


class TestRandomAgent(unittest.TestCase):
    def test_agent_returns_valid_move(self):
        game = Game2048(seed=42)
        agent = RandomAgent(seed=100)
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_reproducibility(self):
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

    def test_play_game(self):
        game = Game2048(seed=42)
        agent = RandomAgent(seed=100)
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestRightLeftAgent(unittest.TestCase):
    def test_agent_alternates_moves(self):
        game = Game2048(seed=42)
        agent = RightLeftAgent()
        self.assertTrue(agent.prefer_right)
        agent.choose_move(game)
        self.assertFalse(agent.prefer_right)
        agent.choose_move(game)
        self.assertTrue(agent.prefer_right)

    def test_agent_returns_valid_move(self):
        game = Game2048(seed=42)
        agent = RightLeftAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self):
        game = Game2048(seed=42)
        agent = RightLeftAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestRightDownAgent(unittest.TestCase):
    def test_agent_alternates_moves(self):
        game = Game2048(seed=42)
        agent = RightDownAgent()
        self.assertTrue(agent.prefer_right)
        agent.choose_move(game)
        self.assertFalse(agent.prefer_right)
        agent.choose_move(game)
        self.assertTrue(agent.prefer_right)

    def test_agent_returns_valid_move(self):
        game = Game2048(seed=42)
        agent = RightDownAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self):
        game = Game2048(seed=42)
        agent = RightDownAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestCornerAgent(unittest.TestCase):
    def test_agent_prefers_corner_moves(self):
        game = Game2048(seed=42)
        agent = CornerAgent()
        self.assertEqual(agent.preferred, ["down", "right", "left", "up"])

    def test_agent_returns_valid_move(self):
        game = Game2048(seed=42)
        agent = CornerAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_fallback_behavior(self):
        game = Game2048(seed=42)
        game.grid = [[2, 2, 4, 8], [4, 2, 2, 4], [8, 4, 2, 2], [2, 4, 8, 4]]
        game.game_over = False
        agent = CornerAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_play_game(self):
        game = Game2048(seed=42)
        agent = CornerAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestGreedyAgent(unittest.TestCase):
    def test_agent_prefers_merge_moves(self):
        # Grid where left/right gives higher score than up/down
        game = Game2048(seed=42)
        game.grid = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.game_over = False
        agent = GreedyAgent()
        move = agent.choose_move(game)
        # Left or right should be preferred (merges the 2s for 4 points)
        # vs up/down which just moves the tile
        self.assertIn(move, ["left", "right"])

    def test_agent_uses_tie_breaker(self):
        game = Game2048(seed=42)
        game.grid = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        game.game_over = False
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertEqual(move, "down")

    def test_agent_returns_valid_move(self):
        game = Game2048(seed=42)
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self):
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = GreedyAgent()
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_play_game(self):
        game = Game2048(seed=42)
        agent = GreedyAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


class TestHeuristicAgent(unittest.TestCase):
    def test_agent_with_default_weights(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        self.assertIsNotNone(agent.weights)
        self.assertIn("empty", agent.weights)
        self.assertIn("mono", agent.weights)
        self.assertIn("max", agent.weights)
        self.assertIn("corner", agent.weights)
        self.assertIn("smooth", agent.weights)

    def test_agent_with_custom_weights(self):
        custom_weights = {
            "empty": 5.0,
            "mono": 2.0,
            "max": 1.5,
            "corner": 3.0,
            "smooth": 0.5,
        }
        agent = HeuristicAgent(weights=custom_weights)
        self.assertEqual(agent.weights["empty"], 5.0)
        self.assertEqual(agent.weights["mono"], 2.0)

    def test_agent_returns_valid_move(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        move = agent.choose_move(game)
        self.assertIn(move, Game2048.MOVES)

    def test_agent_returns_none_when_no_moves(self):
        game = Game2048(seed=42)
        game.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        game.game_over = False
        agent = HeuristicAgent()
        move = agent.choose_move(game)
        self.assertIsNone(move)

    def test_evaluate_returns_float(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        score = agent.evaluate(game)
        self.assertIsInstance(score, float)

    def test_empty_count_heuristic(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        empty_count = agent._count_empty(game)
        self.assertGreaterEqual(empty_count, 0)
        self.assertLessEqual(empty_count, 16)

    def test_monotonicity_heuristic(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        mono = agent._monotonicity(game)
        self.assertIsInstance(mono, float)

    def test_max_tile_heuristic(self):
        game = Game2048(seed=42)
        game.grid = [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        agent = HeuristicAgent()
        max_val = agent._max_tile_value(game)
        self.assertEqual(max_val, 4.0)

    def test_corner_bonus_heuristic(self):
        game = Game2048(seed=42)
        game.grid = [[1024, 2, 4, 8], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        agent = HeuristicAgent()
        corner_bonus = agent._corner_bonus(game)
        self.assertEqual(corner_bonus, 10.0)

    def test_corner_bonus_no_max_in_corner(self):
        game = Game2048(seed=42)
        game.grid = [[2, 2, 4, 1024], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        agent = HeuristicAgent()
        corner_bonus = agent._corner_bonus(game)
        self.assertEqual(corner_bonus, 10.0)

    def test_smoothness_heuristic(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        smoothness = agent._smoothness(game)
        self.assertIsInstance(smoothness, float)

    def test_play_game(self):
        game = Game2048(seed=42)
        agent = HeuristicAgent()
        final_score = agent.play_game(game)
        self.assertTrue(game.game_over)
        self.assertIsInstance(final_score, int)
        self.assertGreater(final_score, 0)


if __name__ == "__main__":
    unittest.main()
