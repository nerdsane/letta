"""
Pricing service for calculating LLM usage costs.

This module provides cost calculation functionality based on token usage and model pricing data.
Pricing data is loaded from pricing_config.json and cached in memory for performance.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from letta.log import get_logger
from letta.schemas.usage import LettaUsageStatistics

logger = get_logger(__name__)


class PricingService:
    """Service for calculating LLM usage costs based on token usage and model pricing."""

    _pricing_config: Optional[Dict] = None
    _config_loaded: bool = False

    @classmethod
    def _load_pricing_config(cls) -> Dict:
        """Load pricing configuration from JSON file. Cached after first load."""
        if cls._config_loaded:
            return cls._pricing_config or {}

        try:
            config_path = Path(__file__).parent / "pricing_config.json"
            with open(config_path, "r") as f:
                cls._pricing_config = json.load(f)
            cls._config_loaded = True
            logger.info(f"Loaded pricing config version {cls._pricing_config.get('version', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to load pricing config: {e}. Cost tracking will be unavailable.")
            cls._pricing_config = {}
            cls._config_loaded = True

        return cls._pricing_config or {}

    @classmethod
    def _get_model_pricing(cls, model: str) -> Optional[Dict]:
        """Get pricing data for a specific model."""
        config = cls._load_pricing_config()
        models = config.get("models", {})

        # Direct match
        if model in models:
            return models[model]

        # Try matching without provider prefix (e.g., "anthropic/claude-sonnet-4.5" -> "claude-sonnet-4.5")
        if "/" in model:
            model_name = model.split("/", 1)[1]
            if model_name in models:
                return models[model_name]

        # Try matching with common variations
        # Remove version suffixes and try again
        base_model = model.split("-")[0:3]  # e.g., "gpt-4o-2024" -> "gpt-4o"
        base_model_str = "-".join(base_model)
        if base_model_str in models:
            return models[base_model_str]

        return None

    @classmethod
    def _calculate_token_cost(cls, tokens: int, price_per_million: float) -> float:
        """Calculate cost for a given number of tokens."""
        return (tokens / 1_000_000) * price_per_million

    @classmethod
    def calculate_cost(cls, usage: LettaUsageStatistics, model: str) -> LettaUsageStatistics:
        """
        Calculate costs for the given usage statistics and model.

        Args:
            usage: Usage statistics with token counts
            model: Model identifier (e.g., "claude-sonnet-4-5-20250929" or "anthropic/claude-sonnet-4-5-20250929")

        Returns:
            Updated usage statistics with cost fields populated (or None if pricing unavailable)
        """
        # Get pricing for this model
        model_pricing = cls._get_model_pricing(model)

        # If no pricing available, return usage unchanged (cost fields will be None)
        if not model_pricing:
            logger.warning(f"No pricing data available for model: {model}")
            return usage

        try:
            # Calculate input cost
            input_cost = None
            if usage.prompt_tokens and usage.prompt_tokens > 0:
                input_price = model_pricing.get("input_token_price", 0)
                input_cost = cls._calculate_token_cost(usage.prompt_tokens, input_price)

            # Calculate output cost
            output_cost = None
            if usage.completion_tokens and usage.completion_tokens > 0:
                output_price = model_pricing.get("output_token_price", 0)
                output_cost = cls._calculate_token_cost(usage.completion_tokens, output_price)

            # Calculate cached input cost (Anthropic-style)
            cached_input_cost = None
            if usage.cached_input_tokens and usage.cached_input_tokens > 0:
                cached_price = model_pricing.get("cached_input_price", 0)
                cached_input_cost = cls._calculate_token_cost(usage.cached_input_tokens, cached_price)

            # Calculate cache write cost (Anthropic-style)
            cache_write_cost = None
            if usage.cache_write_tokens and usage.cache_write_tokens > 0:
                cache_write_price = model_pricing.get("cache_write_price", 0)
                cache_write_cost = cls._calculate_token_cost(usage.cache_write_tokens, cache_write_price)

            # Calculate reasoning cost (for models like o1/o3)
            reasoning_cost = None
            if usage.reasoning_tokens and usage.reasoning_tokens > 0:
                reasoning_price = model_pricing.get("reasoning_token_price", 0)
                reasoning_cost = cls._calculate_token_cost(usage.reasoning_tokens, reasoning_price)

            # Calculate total cost
            total_cost = sum(
                cost for cost in [input_cost, output_cost, cached_input_cost, cache_write_cost, reasoning_cost] if cost is not None
            )

            # Update usage with cost fields
            usage.input_cost = input_cost
            usage.output_cost = output_cost
            usage.cached_input_cost = cached_input_cost
            usage.cache_write_cost = cache_write_cost
            usage.reasoning_cost = reasoning_cost
            usage.total_cost = total_cost if total_cost > 0 else None

            logger.debug(f"Calculated costs for {model}: input=${input_cost:.6f}, output=${output_cost:.6f}, total=${total_cost:.6f}")

        except Exception as e:
            logger.warning(f"Failed to calculate costs for model {model}: {e}")
            # Return usage unchanged if calculation fails

        return usage

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of models with pricing data available."""
        config = cls._load_pricing_config()
        return list(config.get("models", {}).keys())

    @classmethod
    def get_pricing_version(cls) -> Optional[str]:
        """Get the version of the loaded pricing configuration."""
        config = cls._load_pricing_config()
        return config.get("version")
