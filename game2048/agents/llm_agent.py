from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Optional

from openai import OpenAI

from game2048.agents import BaseAgent, register_agent
from game2048.game import Game2048


@register_agent("llm")
class LLMAgent(BaseAgent):
    client: OpenAI
    model: str
    rng: random.Random
    total_cost: float
    total_latency: float
    move_count: int
    latencies: list[float]
    costs: list[float]

    def __init__(
        self,
        model: str = "qwen/qwen3.5-flash-02-23",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        seed: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not found in environment or arguments")

        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.rng = random.Random(seed)
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.move_count = 0
        self.latencies: list[float] = []
        self.costs: list[float] = []

    def _format_board(self, game: Game2048) -> str:
        lines: list[str] = []
        for row in game.grid:
            formatted_row = " ".join(
                f"{tile:5}" if tile > 0 else "    ." for tile in row
            )
            lines.append(formatted_row)
        return "\n".join(lines)

    def _build_prompt(self, game: Game2048) -> str:
        board_str = self._format_board(game)
        valid_moves = game.get_valid_moves()

        prompt = f"""You are playing 2048. Your goal is to combine tiles with the same number to reach 2048.

Current board state (4x4 grid):
{board_str}

Current score: {game.score}
Valid moves: {", ".join(valid_moves)}

Which move should I make next? Respond in JSON format:
{{"move": "up", "reasoning": "brief explanation"}}"""
        return prompt

    def _parse_response(
        self, response_text: str, valid_moves: list[str]
    ) -> Optional[str]:
        text = response_text.strip()

        json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        code_block_matches = re.findall(json_pattern, text)

        for match in code_block_matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict) and "move" in data:
                    move = str(data["move"]).lower()
                    if move in valid_moves:
                        return move
            except json.JSONDecodeError:
                continue

        inline_json_pattern = r"\{[^{}]*\"move\"[^{}]*\}"
        inline_matches = re.findall(inline_json_pattern, text)

        for match in inline_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "move" in data:
                    move = str(data["move"]).lower()
                    if move in valid_moves:
                        return move
            except json.JSONDecodeError:
                continue

        text_lower = text.lower()
        for move in ["up", "down", "left", "right"]:
            if move in text_lower and move in valid_moves:
                return move

        for move in valid_moves:
            if move in text_lower:
                return move

        return None

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * 0.0003
        output_cost = (output_tokens / 1_000_000) * 0.0015
        return input_cost + output_cost

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        prompt = self._build_prompt(game)

        try:
            start_time = time.perf_counter()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )

            latency = time.perf_counter() - start_time

            response_text = response.choices[0].message.content or ""
            parsed_move = self._parse_response(response_text, valid_moves)

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.latencies.append(latency)
            self.costs.append(cost)
            self.total_latency += latency
            self.total_cost += cost
            self.move_count += 1

            if parsed_move:
                return parsed_move

        except Exception:
            pass

        return self.rng.choice(valid_moves)

    def get_stats(self) -> dict[str, float]:
        avg_latency = (
            self.total_latency / self.move_count if self.move_count > 0 else 0.0
        )
        return {
            "total_cost": self.total_cost,
            "total_latency": self.total_latency,
            "move_count": self.move_count,
            "avg_latency": avg_latency,
            "avg_cost": self.total_cost / self.move_count
            if self.move_count > 0
            else 0.0,
        }


@register_agent("llm_history")
class LLMHistoryAgent(LLMAgent):
    state_history: list[tuple[list[list[int]], str, int]]
    history_size: int

    def __init__(
        self,
        model: str = "qwen/qwen3.5-flash-02-23",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        seed: Optional[int] = None,
        history_size: int = 5,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, seed=seed)
        self.history_size = history_size
        self.state_history = []

    def _format_board(self, game: Game2048) -> str:
        return super()._format_board(game)

    def _format_board_from_grid(self, grid: list[list[int]]) -> str:
        lines: list[str] = []
        for row in grid:
            formatted_row = " ".join(
                f"{tile:5}" if tile > 0 else "    ." for tile in row
            )
            lines.append(formatted_row)
        return "\n".join(lines)

    def _build_prompt(self, game: Game2048) -> str:
        board_str = self._format_board(game)
        valid_moves = game.get_valid_moves()

        history_str = ""
        if self.state_history:
            history_lines = []
            for i, (grid, move, score) in enumerate(self.state_history, 1):
                history_lines.append(f"Move {i}:")
                history_lines.append(f"  Board:\n{self._format_board_from_grid(grid)}")
                history_lines.append(f"  Action: {move}")
                history_lines.append(f"  Score: {score}")
                history_lines.append("")
            history_str = "\n".join(history_lines)

        prompt = f"""You are playing 2048. Your goal is to combine tiles with the same number to reach 2048.

Current board state (4x4 grid):
{board_str}

Current score: {game.score}
Valid moves: {", ".join(valid_moves)}
"""

        if history_str:
            prompt += f"""
Recent game history (last {len(self.state_history)} moves):
{history_str}
"""

        prompt += """
Which move should I make next? Respond in JSON format:
{"move": "up", "reasoning": "brief explanation"}"""

        return prompt

    def choose_move(self, game: Game2048) -> Optional[str]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        prompt = self._build_prompt(game)

        try:
            start_time = time.perf_counter()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )

            latency = time.perf_counter() - start_time

            response_text = response.choices[0].message.content or ""
            parsed_move = self._parse_response(response_text, valid_moves)

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.latencies.append(latency)
            self.costs.append(cost)
            self.total_latency += latency
            self.total_cost += cost
            self.move_count += 1

            if parsed_move:
                self.state_history.append(
                    ([row[:] for row in game.grid], parsed_move, game.score)
                )
                if len(self.state_history) > self.history_size:
                    self.state_history = self.state_history[-self.history_size :]
                return parsed_move

        except Exception:
            pass

        fallback_move = self.rng.choice(valid_moves)
        self.state_history.append(
            ([row[:] for row in game.grid], fallback_move, game.score)
        )
        if len(self.state_history) > self.history_size:
            self.state_history = self.state_history[-self.history_size :]
        return fallback_move
