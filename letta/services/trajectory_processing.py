"""
LLM-based processing utilities for trajectories.

This module handles:
1. Generating searchable summaries from trajectory data
2. Scoring trajectories based on outcomes
3. Generating embeddings for similarity search
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from letta.constants import MAX_EMBEDDING_DIM
from letta.log import get_logger
from letta.settings import model_settings

logger = get_logger(__name__)


class TrajectoryProcessor:
    """
    Processes trajectories using LLMs to generate summaries, scores, and embeddings.

    This is the core of the trajectory learning system - it transforms raw execution
    traces into searchable, learnable content.
    """

    def __init__(self):
        self.llm_model = "gpt-4o-mini"  # Fast and cheap for utility tasks
        self.embedding_model = "text-embedding-3-small"  # Standard OpenAI embeddings

    async def generate_searchable_summary(self, trajectory_data: Dict) -> str:
        """
        Generate a natural language summary of the trajectory for search and display.

        The summary captures:
        - What the user wanted (intent, themes, goals)
        - How the agent approached it (strategy, decisions, tools used)
        - What happened (outcome, quality signals)

        This summary is embedded and used for similarity search.
        """
        prompt = f"""Analyze this agent trajectory and create a searchable summary.

Trajectory Data:
{json.dumps(trajectory_data, indent=2)}

Generate a 2-3 paragraph summary that captures:
- What the user wanted (intent, themes, specific requests)
- How the agent approached the task (strategy, key decisions, tools/methods used)
- What happened and the outcome (success/failure, quality signals, user feedback)

Write naturally and focus on information useful for finding similar situations later.
The summary will be embedded and used for semantic search.

Keep it concise but information-rich. Include specific details that make this trajectory unique."""

        client = AsyncOpenAI(api_key=model_settings.openai_api_key)

        try:
            response = await client.chat.completions.create(
                model=self.llm_model, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=500  # Deterministic  # ~2-3 paragraphs
            )

            summary = response.choices[0].message.content
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate trajectory summary: {e}")
            # Fallback: create a basic summary from the data
            return self._create_fallback_summary(trajectory_data)

    async def score_trajectory(self, trajectory_data: Dict) -> Tuple[float, str]:
        """
        Score the trajectory's outcome quality (0-1) with reasoning.

        Considers interaction depth, task complexity, problem-solving richness,
        and learning value - not just task completion.

        Returns:
            (score, reasoning) where score is 0-1 and reasoning explains the score
        """
        # Extract key metrics for context
        metadata = trajectory_data.get("metadata", {})
        turns = trajectory_data.get("turns", [])
        tools_used = metadata.get("tools_used", [])

        prompt = f"""Evaluate this agent trajectory and assign a quality score based on its VALUE for continual learning.

Trajectory Data:
{json.dumps(trajectory_data, indent=2)}

SCORING CRITERIA (weighted):

1. INTERACTION DEPTH (35%):
   - Multiple turns with back-and-forth dialogue?
   - Progressive refinement and iteration?
   - User follow-up questions indicating engagement?
   - Deep problem-solving vs simple Q&A?

2. TASK COMPLEXITY (30%):
   - Challenging/novel request vs routine/trivial?
   - Required reasoning, planning, or creativity?
   - Multi-step process vs single action?
   - Obstacles overcome or edge cases handled?

3. TOOL USAGE & CAPABILITIES (20%):
   - Rich use of available tools/functions?
   - Appropriate tool selection for the task?
   - Sophisticated capability demonstration?

4. LEARNING VALUE (15%):
   - Would this be a useful reference case?
   - Demonstrates important patterns or strategies?
   - Representative of valuable agent behavior?
   - Contains insights for similar future tasks?

Respond with JSON:
{{
  "score": 0.0-1.0,
  "reasoning": "2-3 sentence explanation covering: interaction depth, task complexity, and why this score"
}}

SCORING GUIDELINES:
- 0.9-1.0: Exceptional - Multi-turn complex task with deep problem-solving, rich tool usage, high learning value
- 0.7-0.8: Good - Multi-turn or moderately complex, demonstrates valuable patterns, useful reference
- 0.5-0.6: Mediocre - Simple completion, single-turn trivial request, limited learning value
- 0.3-0.4: Poor - Incomplete, low quality, or minimal interaction
- 0.0-0.2: Failed - Task failed, abandoned early, or error-filled execution

PENALIZE:
- Single-turn trivial completions (e.g., "write 30 words" → done) - score ≤0.5
- No tool usage when tools would add value
- Copy-paste responses without reasoning
- Abandoned or incomplete interactions

REWARD:
- Multi-turn collaborative problem-solving
- Creative solutions to complex challenges
- Sophisticated tool orchestration
- Clear demonstration of agent capabilities

Context from metadata:
- Turns: {len(turns)}
- Tools used: {len(tools_used)}
- Message count: {metadata.get("message_count", 0)}

Be strict and calibrated - reserve high scores (>0.7) for genuinely valuable learning examples."""

        client = AsyncOpenAI(api_key=model_settings.openai_api_key)

        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=200,
            )

            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))

            return score, reasoning

        except Exception as e:
            logger.error(f"Failed to score trajectory: {e}")
            # Fallback: use heuristic scoring
            return self._create_fallback_score(trajectory_data)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding from text for similarity search.

        Uses OpenAI's text-embedding-3-small model (1536 dimensions).
        Pads to MAX_EMBEDDING_DIM to match database schema.
        """
        client = AsyncOpenAI(api_key=model_settings.openai_api_key)

        try:
            response = await client.embeddings.create(input=text, model=self.embedding_model)
            embedding = response.data[0].embedding

            # Pad embedding to MAX_EMBEDDING_DIM to match database column
            embedding_array = np.array(embedding)
            padded_embedding = np.pad(
                embedding_array,
                (0, MAX_EMBEDDING_DIM - len(embedding)),
                mode="constant"
            )

            return padded_embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return None to indicate failure - trajectory will be created but not searchable
            return None

    async def process_trajectory(self, trajectory_data: Dict) -> Tuple[str, float, str, Optional[List[float]]]:
        """
        Full processing pipeline: generate summary, score, and embedding.

        Returns:
            (summary, score, reasoning, embedding)
        """
        # Generate summary
        summary = await self.generate_searchable_summary(trajectory_data)

        # Score trajectory
        score, reasoning = await self.score_trajectory(trajectory_data)

        # Generate embedding from summary
        embedding = await self.generate_embedding(summary)

        return summary, score, reasoning, embedding

    def _create_fallback_summary(self, data: Dict) -> str:
        """Create a basic summary when LLM call fails."""
        summary_parts = []

        # Try to extract common fields
        if "input" in data:
            if isinstance(data["input"], dict) and "initial_prompt" in data["input"]:
                summary_parts.append(f"User request: {data['input']['initial_prompt']}")
            elif isinstance(data["input"], str):
                summary_parts.append(f"User request: {data['input']}")

        if "outcome" in data and isinstance(data["outcome"], dict):
            if "completed" in data["outcome"]:
                status = "Completed" if data["outcome"]["completed"] else "Abandoned"
                summary_parts.append(f"Status: {status}")
            if "user_feedback" in data["outcome"]:
                feedback = data["outcome"]["user_feedback"]
                if isinstance(feedback, dict) and "explicit" in feedback:
                    summary_parts.append(f"Feedback: {feedback['explicit']}")
                elif isinstance(feedback, str):
                    summary_parts.append(f"Feedback: {feedback}")

        if not summary_parts:
            summary_parts.append(f"Agent execution trace with {len(json.dumps(data))} bytes")

        return " | ".join(summary_parts)

    def _create_fallback_score(self, data: Dict) -> Tuple[float, str]:
        """Create a heuristic score when LLM call fails."""
        score = 0.5  # Default neutral
        reasons = []

        # Check completion
        if "outcome" in data and isinstance(data["outcome"], dict):
            if data["outcome"].get("completed"):
                score += 0.2
                reasons.append("Task completed")
            else:
                score -= 0.2
                reasons.append("Task not completed")

            # Check for positive feedback
            feedback = data["outcome"].get("user_feedback", {})
            if isinstance(feedback, dict):
                explicit = feedback.get("explicit", "").lower()
                if any(word in explicit for word in ["good", "great", "loved", "excellent", "perfect"]):
                    score += 0.2
                    reasons.append("Positive user feedback")
                elif any(word in explicit for word in ["bad", "poor", "didn't", "not"]):
                    score -= 0.2
                    reasons.append("Negative user feedback")

        # Clamp to valid range
        score = max(0.0, min(1.0, score))

        reasoning = f"Heuristic score based on: {', '.join(reasons) if reasons else 'no clear signals'}"

        return score, reasoning
