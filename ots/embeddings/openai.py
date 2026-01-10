"""
OpenAI embedding provider for OTS semantic search.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """
    Embedding provider using OpenAI's embedding API.

    Requires the openai package: pip install ots[openai]

    Example:
        provider = OpenAIEmbeddingProvider()
        embedding = await provider.embed("search query")

        # Custom model
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI embedding provider.

        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key
        self._client = None
        self._dimension: Optional[int] = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install ots[openai]"
                )
        return self._client

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        client = self._get_client()

        try:
            response = await client.embeddings.create(
                input=text,
                model=self.model,
            )
            embedding = response.data[0].embedding

            # Cache dimension
            if self._dimension is None:
                self._dimension = len(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = self._get_client()

        try:
            response = await client.embeddings.create(
                input=texts,
                model=self.model,
            )

            # Sort by index to ensure order matches input
            embeddings = sorted(response.data, key=lambda x: x.index)

            # Cache dimension
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0].embedding)

            return [e.embedding for e in embeddings]

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.

        Note: Returns default based on model if not yet computed.
        """
        if self._dimension is not None:
            return self._dimension

        # Default dimensions for known models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return model_dimensions.get(self.model, 1536)
