"""
Decision and entity extractor for extracting decisions and entities from agent turns.

Combines:
1. Programmatic extraction from tool calls (free, immediate)
2. LLM extraction from reasoning (required for full decision trace and entities)

In 'full' mode, extracts both:
- Decisions: What the agent chose to do + reasoning + alternatives
- Entities: Things mentioned/referenced in the conversation + reasoning
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from letta.log import get_logger
from letta.trajectories.ots.models import (
    ContentType,
    DecisionType,
    OTSAlternative,
    OTSChoice,
    OTSConsequence,
    OTSDecision,
    OTSDecisionState,
    OTSEntity,
    OTSTurn,
)

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Result of full extraction containing both decisions and entities."""
    decisions: List[OTSDecision]
    entities: List[OTSEntity]


# Prompt for LLM decision extraction
DECISION_EXTRACTION_PROMPT = """Analyze this agent reasoning and extract decision points.

For each decision, identify:
1. What alternatives were explicitly considered
2. What was chosen and the rationale
3. Any stated confidence level

Reasoning:
{reasoning}

Tool calls made in this turn (for reference):
{tool_calls}

Output JSON array:
[{{
  "decision_type": "reasoning_step | tool_selection | parameter_choice",
  "relates_to_tool_call": "tool_name or null",
  "state": {{ "context_summary": "what the agent understood at this point" }},
  "alternatives": {{
    "considered": [
      {{ "action": "...", "rejected_reason": "..." }}
    ]
  }},
  "choice": {{
    "action": "what was decided",
    "rationale": "why this was chosen",
    "confidence": 0.0-1.0 or null
  }}
}}]

Only include EXPLICIT decision points where the agent weighed options.
Do not invent decisions that aren't in the reasoning.
Return empty array [] if no explicit decision points found."""


# Combined prompt for extracting both decisions and entities
FULL_EXTRACTION_PROMPT = """Analyze this agent turn and extract:
1. DECISIONS: Explicit decision points where the agent weighed options
2. ENTITIES: Things mentioned or operated on

## User Message
{user_message}

## Agent Reasoning
{reasoning}

## Tool Calls Made
{tool_calls}

## Output Format
Return a JSON object with two arrays:

{{
  "decisions": [
    {{
      "decision_type": "reasoning_step | tool_selection | parameter_choice",
      "relates_to_tool_call": "tool_name or null",
      "state": {{ "context_summary": "what the agent understood" }},
      "alternatives": {{
        "considered": [{{ "action": "...", "rejected_reason": "..." }}]
      }},
      "choice": {{
        "action": "what was decided",
        "rationale": "why",
        "confidence": 0.0-1.0 or null
      }}
    }}
  ],
  "entities": [
    {{
      "type": "service | file | user | concept | resource | tool | world | story | rule",
      "id": "identifier or descriptive slug",
      "name": "human-readable name",
      "metadata": {{
        "mentioned_in": "user_message | reasoning | tool_call | tool_result",
        "context": "brief description"
      }}
    }}
  ]
}}

Rules:
- Only include EXPLICIT decision points and entities
- Do not invent things not in the text
- Return {{"decisions": [], "entities": []}} if nothing found"""


class DecisionExtractor:
    """
    Extracts decisions from agent turns.

    Combines programmatic extraction (tool calls) with LLM extraction (reasoning).
    """

    def __init__(self, llm_client=None):
        """
        Initialize extractor.

        Args:
            llm_client: Optional LLM client for reasoning extraction.
                       If not provided, only programmatic extraction is done.
        """
        self.llm_client = llm_client

    def extract_from_turn(
        self,
        turn: OTSTurn,
        mode: str = "full",
    ) -> List[OTSDecision]:
        """
        Extract decisions from a turn.

        Args:
            turn: OTS turn to extract decisions from
            mode: Extraction mode
                - "fast": Only programmatic extraction (tool calls)
                - "full": Programmatic + LLM extraction
                - "deferred": Mark for later extraction

        Returns:
            List of extracted decisions
        """
        # Step 1: Programmatic extraction from tool calls (always free)
        tool_decisions = self._extract_from_tool_calls(turn)

        if mode == "fast":
            return tool_decisions

        if mode == "deferred":
            # Mark as not fully extracted
            for d in tool_decisions:
                # Add marker in state
                if d.state is None:
                    d.state = OTSDecisionState(context_summary=None, available_actions=[])
            return tool_decisions

        # Step 2: LLM extraction from reasoning (if available)
        if self.llm_client and mode == "full":
            reasoning = self._extract_reasoning_text(turn)
            if reasoning:
                return self._extract_with_llm(turn, tool_decisions, reasoning)

        return tool_decisions

    def _extract_from_tool_calls(self, turn: OTSTurn) -> List[OTSDecision]:
        """
        Extract decisions from tool calls (programmatic, no LLM needed).

        Tool calls are explicit decisions - we know:
        - What action was taken (tool name)
        - What arguments were used
        - What the result was (success/failure)

        We don't know (requires LLM):
        - What alternatives were considered
        - Why this tool was chosen
        - The agent's confidence
        """
        decisions = []

        for msg in turn.messages:
            if msg.content.type.value != "tool_call":
                continue

            tool_calls = msg.content.data.get("tool_calls", []) if msg.content.data else []

            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                arguments = func.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except:
                        arguments = {"raw": arguments}

                # Find consequence from tool response
                consequence = self._find_tool_consequence(turn, tc.get("id"))

                decision = OTSDecision(
                    decision_id=f"t{turn.turn_id}-tc-{len(decisions)}",
                    decision_type=DecisionType.TOOL_SELECTION,
                    state=OTSDecisionState(
                        context_summary=None,  # Requires LLM
                        available_actions=[],  # Could populate from tool list
                    ),
                    alternatives=None,  # Requires LLM
                    choice=OTSChoice(
                        action=tool_name,
                        arguments=arguments,
                        rationale=None,  # Requires LLM
                        confidence=None,
                    ),
                    consequence=consequence,
                    evaluation=None,
                    credit_assignment=None,
                )

                decisions.append(decision)

        return decisions

    def _find_tool_consequence(
        self,
        turn: OTSTurn,
        tool_call_id: Optional[str],
    ) -> OTSConsequence:
        """Find the consequence of a tool call from the tool response."""
        if not tool_call_id:
            return OTSConsequence(success=True, result_summary=None, error_type=None)

        for msg in turn.messages:
            if msg.content.type.value != "tool_response":
                continue

            data = msg.content.data or {}
            if data.get("tool_call_id") == tool_call_id:
                text = msg.content.text or ""

                # Check for error indicators
                is_error = any(indicator in text.lower() for indicator in [
                    "error", "exception", "failed", "failure", "invalid"
                ])

                return OTSConsequence(
                    success=not is_error,
                    result_summary=text[:500] if text else None,
                    error_type="tool_error" if is_error else None,
                )

        return OTSConsequence(success=True, result_summary=None, error_type=None)

    def _extract_reasoning_text(self, turn: OTSTurn) -> Optional[str]:
        """Extract reasoning text from assistant messages."""
        reasoning_parts = []

        for msg in turn.messages:
            if msg.reasoning:
                reasoning_parts.append(msg.reasoning)

            # Also check for thinking in content
            if msg.content.text and "<thinking>" in msg.content.text.lower():
                # Extract thinking block
                text = msg.content.text
                start = text.lower().find("<thinking>")
                end = text.lower().find("</thinking>")
                if start != -1 and end != -1:
                    reasoning_parts.append(text[start + 10:end])

        return "\n\n".join(reasoning_parts) if reasoning_parts else None

    async def _extract_with_llm(
        self,
        turn: OTSTurn,
        tool_decisions: List[OTSDecision],
        reasoning: str,
    ) -> List[OTSDecision]:
        """
        Use LLM to extract decisions from reasoning and enrich tool decisions.

        This:
        1. Extracts pure reasoning decisions (not tied to tool calls)
        2. Enriches tool decisions with rationale and alternatives
        """
        if not self.llm_client:
            return tool_decisions

        # Build prompt
        tool_names = [d.choice.action for d in tool_decisions]
        prompt = DECISION_EXTRACTION_PROMPT.format(
            reasoning=reasoning,
            tool_calls=json.dumps(tool_names),
        )

        try:
            # Call LLM
            response = await self.llm_client.generate(
                prompt=prompt,
                response_format="json",
            )

            # Parse response
            extracted = json.loads(response)
            if not isinstance(extracted, list):
                extracted = []

            # Merge with tool decisions
            return self._merge_decisions(tool_decisions, extracted, turn.turn_id)

        except Exception as e:
            logger.warning(f"LLM decision extraction failed: {e}")
            return tool_decisions

    def _merge_decisions(
        self,
        tool_decisions: List[OTSDecision],
        llm_extracted: List[Dict[str, Any]],
        turn_id: int,
    ) -> List[OTSDecision]:
        """
        Merge LLM-extracted decisions with programmatic tool decisions.

        - Tool decisions are enriched with rationale/alternatives from LLM
        - Pure reasoning decisions are added as new entries
        """
        merged = []

        # Process tool decisions
        for td in tool_decisions:
            # Find matching LLM extraction
            matching = None
            for ext in llm_extracted:
                if ext.get("relates_to_tool_call") == td.choice.action:
                    matching = ext
                    break

            if matching:
                # Enrich with LLM data
                td.state = OTSDecisionState(
                    context_summary=matching.get("state", {}).get("context_summary"),
                    available_actions=[],
                )

                alt_data = matching.get("alternatives", {}).get("considered", [])
                if alt_data:
                    td.alternatives = {
                        "considered": [
                            OTSAlternative(
                                action=a.get("action", ""),
                                rationale=a.get("rationale"),
                                rejected_reason=a.get("rejected_reason"),
                            )
                            for a in alt_data
                        ]
                    }

                choice_data = matching.get("choice", {})
                td.choice.rationale = choice_data.get("rationale")
                td.choice.confidence = choice_data.get("confidence")

            merged.append(td)

        # Add pure reasoning decisions
        for ext in llm_extracted:
            if ext.get("relates_to_tool_call") is None:
                decision_type = ext.get("decision_type", "reasoning_step")
                if decision_type == "reasoning_step":
                    dt = DecisionType.REASONING_STEP
                elif decision_type == "parameter_choice":
                    dt = DecisionType.PARAMETER_CHOICE
                else:
                    dt = DecisionType.REASONING_STEP

                choice_data = ext.get("choice", {})
                alt_data = ext.get("alternatives", {}).get("considered", [])

                decision = OTSDecision(
                    decision_id=f"t{turn_id}-rs-{len(merged)}",
                    decision_type=dt,
                    state=OTSDecisionState(
                        context_summary=ext.get("state", {}).get("context_summary"),
                        available_actions=[],
                    ),
                    alternatives={
                        "considered": [
                            OTSAlternative(
                                action=a.get("action", ""),
                                rationale=a.get("rationale"),
                                rejected_reason=a.get("rejected_reason"),
                            )
                            for a in alt_data
                        ]
                    } if alt_data else None,
                    choice=OTSChoice(
                        action=choice_data.get("action", ""),
                        arguments=None,
                        rationale=choice_data.get("rationale"),
                        confidence=choice_data.get("confidence"),
                    ),
                    consequence=OTSConsequence(
                        success=True,  # Reasoning decisions don't fail
                        result_summary=None,
                        error_type=None,
                    ),
                    evaluation=None,
                    credit_assignment=None,
                )

                merged.append(decision)

        return merged

    def extract_from_turn_sync(
        self,
        turn: OTSTurn,
        mode: str = "fast",
    ) -> List[OTSDecision]:
        """
        Synchronous version of extract_from_turn.

        Only supports "fast" mode (programmatic extraction).
        Use async version for LLM extraction.
        """
        if mode != "fast":
            logger.warning("Sync extraction only supports 'fast' mode. Use async for full extraction.")
            mode = "fast"

        return self.extract_from_turn(turn, mode=mode)

    async def extract_full(self, turn: OTSTurn) -> ExtractionResult:
        """
        Extract both decisions and entities from a turn using LLM.

        This is the recommended method for full OTS extraction with rich data:
        - Decisions with rationale, alternatives, confidence
        - Entities (services, files, users, concepts, resources)

        Args:
            turn: OTS turn to extract from

        Returns:
            ExtractionResult containing decisions and entities
        """
        # Start with programmatic extraction
        tool_decisions = self._extract_from_tool_calls(turn)

        if not self.llm_client:
            logger.warning("No LLM client provided. Returning programmatic extraction only.")
            return ExtractionResult(decisions=tool_decisions, entities=[])

        # Get reasoning and user message for context
        reasoning = self._extract_reasoning_text(turn) or ""
        user_message = self._extract_user_message(turn) or ""
        tool_names = [d.choice.action for d in tool_decisions]

        # Build prompt
        prompt = FULL_EXTRACTION_PROMPT.format(
            user_message=user_message,
            reasoning=reasoning,
            tool_calls=json.dumps(tool_names),
        )

        try:
            # Call LLM for full extraction
            response = await self.llm_client.generate(
                prompt=prompt,
                response_format="json",
            )

            # Parse response
            extracted = json.loads(response)
            if not isinstance(extracted, dict):
                logger.warning("LLM returned non-dict response for full extraction")
                return ExtractionResult(decisions=tool_decisions, entities=[])

            # Extract decisions and entities from response
            llm_decisions = extracted.get("decisions", [])
            llm_entities = extracted.get("entities", [])

            # Merge LLM decisions with tool decisions
            merged_decisions = self._merge_decisions(tool_decisions, llm_decisions, turn.turn_id)

            # Convert entity dicts to OTSEntity objects
            entities = self._parse_entities(llm_entities)

            return ExtractionResult(decisions=merged_decisions, entities=entities)

        except Exception as e:
            logger.warning(f"LLM full extraction failed: {e}")
            return ExtractionResult(decisions=tool_decisions, entities=[])

    def _extract_user_message(self, turn: OTSTurn) -> Optional[str]:
        """Extract user message text from turn."""
        for msg in turn.messages:
            if msg.role == "user" and msg.content.text:
                return msg.content.text
        return None

    def _parse_entities(self, entity_dicts: List[Dict[str, Any]]) -> List[OTSEntity]:
        """Parse entity dictionaries into OTSEntity objects."""
        entities = []
        for e in entity_dicts:
            try:
                entity = OTSEntity(
                    type=e.get("type", "concept"),
                    id=e.get("id", ""),
                    name=e.get("name"),
                    metadata=e.get("metadata", {}),
                )
                entities.append(entity)
            except Exception as ex:
                logger.warning(f"Failed to parse entity: {e}, error: {ex}")
        return entities
