#!/bin/bash
# Example commands for running LLM model evaluation
# Note: Requires OPENROUTER_API_KEY environment variable

# Quick test with 2 models, 3 games each (V1 only)
# Estimated time: ~10-15 minutes
# Estimated cost: ~$0.01-0.05
uv run --env-file ~/.env.openrouter python scripts/evaluate_llm_models.py \
  --models "qwen/qwen3.5-35b-a3b" "z-ai/glm-5" \
  --games 3 \
  --v1-only \
  --verbose

# Test all 6 models with 3 games each (both V1 and V2)
# Recommended starting point for testing
# Estimated time: ~30-45 minutes
# Estimated cost: ~$0.05-0.20
uv run --env-file ~/.env.openrouter python scripts/evaluate_llm_models.py \
  --models \
    "qwen/qwen3.5-35b-a3b" \
    "qwen/qwen3.5-27b" \
    "qwen/qwen3.5-122b-a10b" \
    "qwen/qwen3.5-397b-a17b" \
    "z-ai/glm-5" \
    "moonshotai/kimi-k2.5" \
  --games 3 \
  --verbose

# Full benchmark: all 6 models, 10 games each (both V1 and V2)
# This is the complete evaluation as specified in Issue #56
# Estimated time: ~2-3 hours
# Estimated cost: ~$0.50-2.00
uv run --env-file ~/.env.openrouter python scripts/evaluate_llm_models.py \
  --models \
    "qwen/qwen3.5-35b-a3b" \
    "qwen/qwen3.5-27b" \
    "qwen/qwen3.5-122b-a10b" \
    "qwen/qwen3.5-397b-a17b" \
    "z-ai/glm-5" \
    "moonshotai/kimi-k2.5" \
  --games 10 \
  --verbose

# Test only V2 agents with custom history size
uv run --env-file ~/.env.openrouter python scripts/evaluate_llm_models.py \
  --models "qwen/qwen3.5-35b-a3b" "z-ai/glm-5" \
  --games 5 \
  --v2-only \
  --history-size 10 \
  --verbose

# Custom output directory
uv run --env-file ~/.env.openrouter python scripts/evaluate_llm_models.py \
  --models "qwen/qwen3.5-35b-a3b" \
  --games 3 \
  --output-dir ./my_evaluation \
  --verbose
