"""
Decision embedding utilities for OTS trajectories.

Provides semantic embedding generation for individual decisions,
enabling decision-level similarity search for context learning.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from letta.constants import MAX_EMBEDDING_DIM
from letta.log import get_logger
from letta.trajectories.ots.models import (
    OTSDecision,
    OTSTrajectory,
    OTSTurn,
)

logger = get_logger(__name__)


class DecisionEmbedder:
    """
    Generates embeddings for OTS decisions.

    Decisions are embedded based on:
    - Decision state (context summary, available actions)
    - Choice (action, arguments, rationale)
    - Consequence (success, result summary)

    This enables retrieval of similar decision points for context learning.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize decision embedder.

        Args:
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model

    async def embed_decision(self, decision: OTSDecision) -> Optional[List[float]]:
        """
        Generate embedding for a single decision.

        Args:
            decision: OTS decision to embed

        Returns:
            Embedding vector or None on failure
        """
        text = self._decision_to_text(decision)
        return await self._generate_embedding(text)

    async def embed_decisions(
        self,
        decisions: List[OTSDecision],
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple decisions.

        Args:
            decisions: List of OTS decisions

        Returns:
            List of embeddings (None for failures)
        """
        texts = [self._decision_to_text(d) for d in decisions]
        return await self._generate_embeddings_batch(texts)

    async def embed_trajectory_decisions(
        self,
        trajectory: OTSTrajectory,
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for all decisions in a trajectory.

        Args:
            trajectory: OTS trajectory containing decisions

        Returns:
            Dict mapping decision_id -> embedding
        """
        embeddings = {}

        for turn in trajectory.turns:
            for decision in turn.decisions:
                embedding = await self.embed_decision(decision)
                if embedding:
                    embeddings[decision.decision_id] = embedding

        return embeddings

    def _decision_to_text(self, decision: OTSDecision) -> str:
        """
        Convert a decision to text for embedding.

        The text representation captures the key decision elements:
        - Context/state at decision point
        - Action taken and reasoning
        - Outcome/consequence
        """
        parts = []

        # Decision type
        parts.append(f"Decision type: {decision.decision_type.value}")

        # State/context
        if decision.state:
            if decision.state.context_summary:
                parts.append(f"Context: {decision.state.context_summary}")
            if decision.state.available_actions:
                parts.append(f"Available actions: {', '.join(decision.state.available_actions)}")

        # Choice
        parts.append(f"Action: {decision.choice.action}")
        if decision.choice.arguments:
            # Summarize arguments (avoid embedding large payloads)
            args_summary = self._summarize_arguments(decision.choice.arguments)
            parts.append(f"Arguments: {args_summary}")
        if decision.choice.rationale:
            parts.append(f"Rationale: {decision.choice.rationale}")

        # Consequence
        outcome = "Success" if decision.consequence.success else "Failure"
        parts.append(f"Outcome: {outcome}")
        if decision.consequence.result_summary:
            parts.append(f"Result: {decision.consequence.result_summary[:200]}")

        # Alternatives considered
        if decision.alternatives and decision.alternatives.get("considered"):
            alts = decision.alternatives["considered"]
            alt_names = [a.action for a in alts[:3]]  # Top 3
            parts.append(f"Alternatives considered: {', '.join(alt_names)}")

        return " | ".join(parts)

    def _summarize_arguments(self, arguments: Dict[str, Any], max_length: int = 100) -> str:
        """Summarize arguments for embedding, keeping key info."""
        try:
            if not arguments:
                return ""

            # Extract key argument names and types
            summary_parts = []
            for key, value in list(arguments.items())[:5]:  # Top 5 args
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                elif isinstance(value, (dict, list)):
                    value = f"<{type(value).__name__}>"
                summary_parts.append(f"{key}={value}")

            result = ", ".join(summary_parts)
            return result[:max_length] if len(result) > max_length else result
        except Exception:
            return ""

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI."""
        try:
            import openai

            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding

            # Pad to MAX_EMBEDDING_DIM
            embedding_array = np.array(embedding)
            padded_embedding = np.pad(
                embedding_array,
                (0, MAX_EMBEDDING_DIM - len(embedding)),
                mode="constant",
                constant_values=0,
            )

            return padded_embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate decision embedding: {e}")
            return None

    async def _generate_embeddings_batch(
        self,
        texts: List[str],
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in a batch."""
        try:
            import openai

            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )

            results = []
            for i, text in enumerate(texts):
                try:
                    embedding = response.data[i].embedding
                    embedding_array = np.array(embedding)
                    padded_embedding = np.pad(
                        embedding_array,
                        (0, MAX_EMBEDDING_DIM - len(embedding)),
                        mode="constant",
                        constant_values=0,
                    )
                    results.append(padded_embedding.tolist())
                except Exception:
                    results.append(None)

            return results

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)

    # Handle zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class DecisionSearcher:
    """
    Search for similar decisions across trajectories.

    Uses decision embeddings for semantic similarity matching.
    This enables context learning by finding relevant past decisions.
    """

    def __init__(self, embedder: Optional[DecisionEmbedder] = None):
        """
        Initialize decision searcher.

        Args:
            embedder: Decision embedder instance (creates new if None)
        """
        self.embedder = embedder or DecisionEmbedder()

    async def search_similar_decisions(
        self,
        query_decision: OTSDecision,
        candidate_decisions: List[Tuple[OTSDecision, List[float]]],
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Tuple[OTSDecision, float]]:
        """
        Search for decisions similar to a query decision.

        Args:
            query_decision: Decision to find similar matches for
            candidate_decisions: List of (decision, embedding) tuples to search
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (decision, similarity_score) tuples, sorted by similarity
        """
        # Embed query decision
        query_embedding = await self.embedder.embed_decision(query_decision)
        if not query_embedding:
            return []

        # Calculate similarities
        similarities = []
        for decision, embedding in candidate_decisions:
            if embedding:
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= min_similarity:
                    similarities.append((decision, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def search_by_context(
        self,
        context_description: str,
        candidate_decisions: List[Tuple[OTSDecision, List[float]]],
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Tuple[OTSDecision, float]]:
        """
        Search for decisions relevant to a context description.

        Args:
            context_description: Natural language description of the context
            candidate_decisions: List of (decision, embedding) tuples to search
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (decision, similarity_score) tuples
        """
        # Embed context description
        query_embedding = await self.embedder._generate_embedding(context_description)
        if not query_embedding:
            return []

        # Calculate similarities
        similarities = []
        for decision, embedding in candidate_decisions:
            if embedding:
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= min_similarity:
                    similarities.append((decision, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


# Convenience functions

async def embed_decision(decision: OTSDecision) -> Optional[List[float]]:
    """Embed a single decision."""
    embedder = DecisionEmbedder()
    return await embedder.embed_decision(decision)


async def find_similar_decisions(
    query: str,
    decisions_with_embeddings: List[Tuple[OTSDecision, List[float]]],
    top_k: int = 5,
) -> List[Tuple[OTSDecision, float]]:
    """Find decisions similar to a query string."""
    searcher = DecisionSearcher()
    return await searcher.search_by_context(
        query,
        decisions_with_embeddings,
        top_k=top_k,
    )
