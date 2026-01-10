"""
Embedding providers for OTS semantic search.

Provides:
- OpenAIEmbeddingProvider: Default embedding provider using OpenAI
"""

from ots.embeddings.openai import OpenAIEmbeddingProvider

__all__ = [
    "OpenAIEmbeddingProvider",
]
