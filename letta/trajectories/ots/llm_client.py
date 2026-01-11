"""
OpenAI LLM client adapter for OTS decision/entity extraction.

Implements the OTS LLMClient protocol using OpenAI's API.
Uses GPT-5 mini for optimal balance of reasoning capability and cost.
"""

from typing import Optional

from openai import AsyncOpenAI

from letta.log import get_logger
from letta.settings import model_settings

logger = get_logger(__name__)


class OpenAILLMClient:
    """
    OpenAI LLM client implementing the OTS LLMClient protocol.

    Used by DecisionExtractor to extract rationale, alternatives,
    and entities from agent reasoning using LLM analysis.

    Example:
        client = OpenAILLMClient()
        extractor = DecisionExtractor(llm_client=client)
        result = await extractor.extract_full(turn)
    """

    # Default model for OTS extraction
    # GPT-4o mini: Fast, cost-effective, good for extraction
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Initialize OpenAI LLM client.

        Args:
            model: Model to use (default: gpt-5-mini)
            api_key: OpenAI API key (default: from settings)
            temperature: Sampling temperature (default: 0.3 for consistency)
            max_tokens: Max output tokens (default: 2000)
        """
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use provided API key or fall back to settings
        api_key = api_key or model_settings.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI LLM client with model: {self.model}")

    async def generate(
        self,
        prompt: str,
        response_format: str = "text",
    ) -> str:
        """
        Generate text from prompt.

        Implements the OTS LLMClient protocol.

        Args:
            prompt: Prompt to send to LLM
            response_format: "text" or "json"

        Returns:
            Generated text response
        """
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Request JSON output format if specified
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if content is None:
                logger.warning("LLM returned empty response")
                return "{}" if response_format == "json" else ""

            return content

        except Exception as e:
            logger.error(f"OpenAI LLM generation failed: {e}")
            raise


class AnthropicLLMClient:
    """
    Anthropic LLM client implementing the OTS LLMClient protocol.

    Alternative to OpenAI for users who prefer Claude models.
    Uses Claude Haiku 4.5 for cost-effective extraction.

    Example:
        client = AnthropicLLMClient()
        extractor = DecisionExtractor(llm_client=client)
        result = await extractor.extract_full(turn)
    """

    # Claude Haiku 4.5: $1/1M input, $5/1M output
    # Excellent reasoning, 73.3% SWE-bench
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Initialize Anthropic LLM client.

        Args:
            model: Model to use (default: claude-haiku-4-5)
            api_key: Anthropic API key (default: from settings)
            temperature: Sampling temperature (default: 0.3 for consistency)
            max_tokens: Max output tokens (default: 2000)
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for AnthropicLLMClient. "
                "Install with: pip install anthropic"
            )

        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use provided API key or fall back to settings
        api_key = api_key or getattr(model_settings, "anthropic_api_key", None)
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = AsyncAnthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic LLM client with model: {self.model}")

    async def generate(
        self,
        prompt: str,
        response_format: str = "text",
    ) -> str:
        """
        Generate text from prompt.

        Implements the OTS LLMClient protocol.

        Args:
            prompt: Prompt to send to LLM
            response_format: "text" or "json"

        Returns:
            Generated text response
        """
        # For JSON format, add instruction to the prompt
        if response_format == "json":
            prompt = f"{prompt}\n\nRespond with valid JSON only."

        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract text from response
            content = response.content[0].text if response.content else ""

            if not content:
                logger.warning("Anthropic LLM returned empty response")
                return "{}" if response_format == "json" else ""

            return content

        except Exception as e:
            logger.error(f"Anthropic LLM generation failed: {e}")
            raise


def get_default_llm_client() -> OpenAILLMClient:
    """
    Get the default LLM client for OTS extraction.

    Returns OpenAI client with GPT-5 mini by default.

    Returns:
        Configured LLM client
    """
    return OpenAILLMClient()
