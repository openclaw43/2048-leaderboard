#!/usr/bin/env python3
"""Test the evaluate_llm_models script logic without API calls."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_llm_models import generate_comparison_table


def test_comparison_table_generation() -> None:
    mock_results = [
        {
            "model": "qwen/qwen3.5-35b-a3b",
            "agent_type": "V1",
            "num_games": 3,
            "avg_score": 2500.5,
            "median_score": 2400,
            "max_score": 3000,
            "min_score": 2100,
            "avg_latency": 0.5,
            "avg_cost": 0.001,
            "avg_moves": 150.5,
            "total_cost": 0.003,
            "total_moves": 452,
            "cost_per_game": 0.001,
            "scores": [2400, 2100, 3000],
            "max_tiles": [512, 256, 1024],
            "games": [],
        },
        {
            "model": "qwen/qwen3.5-35b-a3b",
            "agent_type": "V2",
            "num_games": 3,
            "avg_score": 2800.3,
            "median_score": 2700,
            "max_score": 3200,
            "min_score": 2500,
            "avg_latency": 0.7,
            "avg_cost": 0.0015,
            "avg_moves": 160.2,
            "total_cost": 0.0045,
            "total_moves": 481,
            "cost_per_game": 0.0015,
            "scores": [2700, 2500, 3200],
            "max_tiles": [512, 512, 1024],
            "games": [],
        },
        {
            "model": "z-ai/glm-5",
            "agent_type": "V1",
            "num_games": 3,
            "avg_score": 2300.0,
            "median_score": 2300,
            "max_score": 2500,
            "min_score": 2100,
            "avg_latency": 0.6,
            "avg_cost": 0.0012,
            "avg_moves": 140.0,
            "total_cost": 0.0036,
            "total_moves": 420,
            "cost_per_game": 0.0012,
            "scores": [2300, 2100, 2500],
            "max_tiles": [512, 256, 512],
            "games": [],
        },
        {
            "model": "z-ai/glm-5",
            "agent_type": "V2",
            "num_games": 3,
            "avg_score": 2600.0,
            "median_score": 2600,
            "max_score": 2800,
            "min_score": 2400,
            "avg_latency": 0.8,
            "avg_cost": 0.0018,
            "avg_moves": 155.0,
            "total_cost": 0.0054,
            "total_moves": 465,
            "cost_per_game": 0.0018,
            "scores": [2600, 2400, 2800],
            "max_tiles": [512, 512, 512],
            "games": [],
        },
    ]

    table = generate_comparison_table(mock_results)

    print("=" * 60)
    print("GENERATED COMPARISON TABLE")
    print("=" * 60)
    print(table)
    print("=" * 60)

    assert "# LLM Model Comparison" in table
    assert "qwen3.5-35b-a3b" in table
    assert "glm-5" in table
    assert "V1" in table
    assert "V2" in table
    assert "2500.5" in table
    assert "2300.0" in table
    assert "Best V1 Score" in table
    assert "Best V2 Score" in table
    assert "Cost Analysis" in table

    print("\n✓ All assertions passed!")
    print("✓ Comparison table generated successfully!")


if __name__ == "__main__":
    test_comparison_table_generation()
