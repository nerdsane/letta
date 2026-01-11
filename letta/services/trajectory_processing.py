"""
LLM-based processing utilities for trajectories.

This module handles:
1. Generating searchable summaries from trajectory data
2. Scoring trajectories based on outcomes
3. Generating embeddings for similarity search
4. OTS decision/entity extraction with LLM enrichment
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from letta.constants import MAX_EMBEDDING_DIM
from letta.log import get_logger
from letta.settings import model_settings
from letta.trajectories.ots.adapter import OTSAdapter
from letta.trajectories.ots.decision_extractor import DecisionExtractor
from letta.trajectories.ots.dsf_entity_extractor import DSFEntityExtractor
from letta.trajectories.ots.llm_client import OpenAILLMClient

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

    async def extract_labels_and_metadata(self, trajectory_data: Dict) -> Tuple[List[str], str, str, Dict]:
        """
        Extract structured labels and metadata from trajectory for pattern detection and filtering.

        Returns:
            (tags, task_category, complexity_level, trajectory_metadata)
            - tags: List of semantic labels like ['creative', 'analytical', 'iterative']
            - task_category: Primary classification like 'code_generation', 'debugging', 'research'
            - complexity_level: One of 'trivial', 'simple', 'moderate', 'complex', 'expert'
            - trajectory_metadata: Dict with flexible additional info (interaction patterns, tool usage, etc.)
        """
        prompt = f"""Analyze this agent trajectory and extract structured metadata for categorization and pattern detection.

Trajectory Data:
{json.dumps(trajectory_data, indent=2)}

Extract the following structured metadata:

1. TAGS (3-7 semantic labels):
   - Descriptive labels that capture the nature of the work
   - Examples: 'creative', 'analytical', 'iterative', 'collaborative', 'technical',
     'debugging', 'research', 'problem_solving', 'code_generation', 'informational'
   - Focus on characteristics that would help find similar trajectories

2. TASK_CATEGORY (single primary classification):
   - The main type of work being done
   - Examples: 'code_generation', 'debugging', 'research', 'data_analysis', 'documentation',
     'refactoring', 'design', 'testing', 'content_creation', 'information_retrieval'
   - Choose the ONE category that best represents the overall task

3. COMPLEXITY_LEVEL (single value):
   - Overall difficulty/sophistication level
   - One of: 'trivial', 'simple', 'moderate', 'complex', 'expert'
   - Consider: reasoning depth, tool orchestration, problem difficulty, multi-step planning

4. METADATA (flexible dictionary with patterns and insights):
   - interaction_style: 'exploratory' | 'directive' | 'collaborative' | 'iterative'
   - tool_orchestration: 'single_tool' | 'coordinated_tools' | 'advanced_workflow'
   - problem_solving_approach: brief description of strategy used
   - key_capabilities_demonstrated: list of notable skills/features shown
   - any other interesting patterns you notice

Respond with JSON:
{{
  "tags": ["tag1", "tag2", "tag3"],
  "task_category": "category",
  "complexity_level": "moderate",
  "metadata": {{
    "interaction_style": "...",
    "tool_orchestration": "...",
    "problem_solving_approach": "...",
    "key_capabilities_demonstrated": ["...", "..."]
  }}
}}

Be specific and descriptive. These labels will be used for filtering, pattern detection, and dashboard analytics."""

        client = AsyncOpenAI(api_key=model_settings.openai_api_key)

        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )

            result = json.loads(response.choices[0].message.content)

            tags = result.get("tags", [])
            task_category = result.get("task_category", "unknown")
            complexity_level = result.get("complexity_level", "moderate")
            metadata = result.get("metadata", {})

            # Validate complexity_level
            valid_levels = ["trivial", "simple", "moderate", "complex", "expert"]
            if complexity_level not in valid_levels:
                complexity_level = "moderate"

            return tags, task_category, complexity_level, metadata

        except Exception as e:
            logger.error(f"Failed to extract labels and metadata: {e}")
            # Fallback to basic extraction
            return self._create_fallback_labels(trajectory_data)

    async def extract_ots_decisions_and_entities(
        self,
        trajectory_data: Dict,
        use_llm: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract OTS-style decisions and entities from trajectory.

        Uses OTS's DecisionExtractor with optional LLM enrichment for:
        - Decisions with rationale, alternatives, confidence
        - Entities (services, files, users, concepts, resources)

        Args:
            trajectory_data: Raw trajectory data dict
            use_llm: Whether to use LLM for rich extraction (default: True)

        Returns:
            (decisions, entities) as lists of dicts
        """
        try:
            # Convert run data to OTS format
            ots_trajectory = OTSAdapter.from_letta_run(trajectory_data)

            # Create extractor with optional LLM client
            llm_client = OpenAILLMClient() if use_llm else None
            extractor = DecisionExtractor(llm_client=llm_client)

            # Extract decisions and entities from each turn
            all_decisions = []
            all_entities = []

            for turn in ots_trajectory.turns:
                if use_llm and llm_client:
                    # Full extraction with LLM (rich data)
                    result = await extractor.extract_full(turn)
                    all_decisions.extend([self._decision_to_dict(d) for d in result.decisions])
                    all_entities.extend([self._entity_to_dict(e) for e in result.entities])
                else:
                    # Fast extraction (programmatic only)
                    decisions = extractor.extract_from_turn_sync(turn, mode="fast")
                    all_decisions.extend([self._decision_to_dict(d) for d in decisions])

            # Also run DSF-specific entity extraction (programmatic, no LLM needed)
            try:
                dsf_extractor = DSFEntityExtractor()
                dsf_entities = dsf_extractor.extract_all(ots_trajectory)
                all_entities.extend([self._entity_to_dict(e) for e in dsf_entities])
            except Exception as dsf_err:
                logger.warning(f"DSF entity extraction failed: {dsf_err}")

            # Deduplicate entities by ID
            seen_ids = set()
            unique_entities = []
            for e in all_entities:
                # OTSEntity uses 'id', not 'entity_id'
                entity_id = e.get("id", "")
                if entity_id and entity_id not in seen_ids:
                    unique_entities.append(e)
                    seen_ids.add(entity_id)

            return all_decisions, unique_entities

        except Exception as e:
            logger.error(f"OTS extraction failed: {e}")
            return [], []

    def _decision_to_dict(self, decision) -> Dict[str, Any]:
        """Convert OTSDecision to serializable dict."""
        try:
            # Use model_dump if available (Pydantic v2)
            if hasattr(decision, "model_dump"):
                return decision.model_dump()
            # Fallback to dict() for dataclass
            elif hasattr(decision, "__dict__"):
                return self._serialize_decision(decision)
            return {}
        except Exception:
            return {}

    def _serialize_decision(self, decision) -> Dict[str, Any]:
        """Manually serialize decision dataclass."""
        return {
            "decision_id": decision.decision_id,
            "decision_type": decision.decision_type.value if hasattr(decision.decision_type, "value") else str(decision.decision_type),
            "state": {
                "context_summary": decision.state.context_summary if decision.state else None,
                "available_actions": decision.state.available_actions if decision.state else [],
            } if decision.state else None,
            "alternatives": decision.alternatives,
            "choice": {
                "action": decision.choice.action,
                "arguments": decision.choice.arguments,
                "rationale": decision.choice.rationale,
                "confidence": decision.choice.confidence,
            } if decision.choice else None,
            "consequence": {
                "success": decision.consequence.success,
                "result_summary": decision.consequence.result_summary,
                "error_type": decision.consequence.error_type,
            } if decision.consequence else None,
            "evaluation": decision.evaluation,
            "credit_assignment": decision.credit_assignment,
        }

    def _entity_to_dict(self, entity) -> Dict[str, Any]:
        """Convert OTSEntity to serializable dict."""
        try:
            if hasattr(entity, "model_dump"):
                return entity.model_dump()
            elif hasattr(entity, "__dict__"):
                # OTSEntity uses 'type' and 'id' as field names
                return {
                    "type": entity.type,
                    "id": entity.id,
                    "name": entity.name,
                    "metadata": entity.metadata,
                }
            return {}
        except Exception as e:
            logger.warning(f"Failed to convert entity to dict: {e}")
            return {}

    async def process_trajectory(
        self,
        trajectory_data: Dict,
        extract_ots: bool = True,
    ) -> Tuple[str, float, str, List[str], str, str, Dict, Optional[List[float]], List[Dict], List[Dict]]:
        """
        Full processing pipeline: generate summary, score, labels, metadata, embedding, and OTS data.

        Args:
            trajectory_data: Raw trajectory data
            extract_ots: Whether to extract OTS decisions/entities with LLM (default: True)

        Returns:
            (summary, score, reasoning, tags, task_category, complexity_level,
             trajectory_metadata, embedding, ots_decisions, ots_entities)
        """
        # Generate summary
        summary = await self.generate_searchable_summary(trajectory_data)

        # Score trajectory
        score, reasoning = await self.score_trajectory(trajectory_data)

        # Extract labels and metadata
        tags, task_category, complexity_level, metadata = await self.extract_labels_and_metadata(trajectory_data)

        # Generate embedding from summary
        embedding = await self.generate_embedding(summary)

        # Extract OTS decisions and entities
        ots_decisions = []
        ots_entities = []
        if extract_ots:
            ots_decisions, ots_entities = await self.extract_ots_decisions_and_entities(
                trajectory_data,
                use_llm=True,
            )

        return summary, score, reasoning, tags, task_category, complexity_level, metadata, embedding, ots_decisions, ots_entities

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

    def _create_fallback_labels(self, data: Dict) -> Tuple[List[str], str, str, Dict]:
        """Create basic labels when LLM call fails."""
        tags = []
        task_category = "unknown"
        complexity_level = "moderate"
        metadata = {}

        # Try to infer from metadata
        traj_metadata = data.get("metadata", {})

        # Count turns as complexity heuristic
        turns = data.get("turns", [])
        if len(turns) <= 2:
            complexity_level = "simple"
        elif len(turns) >= 10:
            complexity_level = "complex"

        # Tools used
        tools_used = traj_metadata.get("tools_used", [])
        if tools_used:
            tags.append("tool_usage")
            metadata["tools_used_count"] = len(tools_used)

        # Message count
        message_count = traj_metadata.get("message_count", 0)
        if message_count > 10:
            tags.append("extended_interaction")

        # Default tags if none found
        if not tags:
            tags = ["general"]

        metadata["fallback"] = True
        metadata["turns_count"] = len(turns)

        return tags, task_category, complexity_level, metadata
