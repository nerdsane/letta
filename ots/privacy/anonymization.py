"""
Core anonymization functions for OTS trajectories.

Provides utilities to create anonymized copies of trajectories
that preserve learning signal while protecting sensitive information.
"""

import hashlib
from typing import Any, Dict, List, Optional, Union

from ots.models import (
    OTSAlternative,
    OTSChoice,
    OTSConsequence,
    OTSContext,
    OTSContextSnapshot,
    OTSCreditAssignment,
    OTSDecision,
    OTSDecisionEvaluation,
    OTSDecisionState,
    OTSEntity,
    OTSMessage,
    OTSMessageContent,
    OTSMetadata,
    OTSResource,
    OTSSystemMessage,
    OTSTrajectory,
    OTSTurn,
    OTSUser,
    OTSVisibility,
)
from ots.privacy.protocols import AnonymizationPolicy, DefaultAnonymizationPolicy


def hash_identifier(identifier: str, salt: str = "") -> str:
    """
    Create deterministic hash of identifier.

    Args:
        identifier: The ID to hash
        salt: Optional salt for organization-specific hashing

    Returns:
        16-character hex hash
    """
    return hashlib.sha256(f"{salt}{identifier}".encode()).hexdigest()[:16]


def anonymize_trajectory(
    trajectory: OTSTrajectory,
    policy: Optional[AnonymizationPolicy] = None,
    salt: str = "",
) -> OTSTrajectory:
    """
    Create anonymized copy of trajectory preserving learning signal.

    What is preserved (for learning value):
    - Structure: turn sequence, decision types, message roles
    - Timing: timestamps, durations
    - Outcomes: success/failure, error types, scores
    - Patterns: action names, decision types, rationale summaries

    What is anonymized:
    - IDs: hashed with deterministic algorithm
    - Content: message text, tool arguments, results
    - Context: user info, referrers, URIs
    - Reasoning: full chain-of-thought (summarized if learning policy)

    Args:
        trajectory: The trajectory to anonymize
        policy: Anonymization policy (defaults to DefaultAnonymizationPolicy)
        salt: Salt for ID hashing (use per-org for cross-org sharing)

    Returns:
        Anonymized copy of the trajectory
    """
    policy = policy or DefaultAnonymizationPolicy()

    return OTSTrajectory(
        trajectory_id=hash_identifier(trajectory.trajectory_id, salt),
        version=trajectory.version,
        metadata=_anonymize_metadata(trajectory.metadata, policy, salt),
        context=_anonymize_context(trajectory.context, policy, salt),
        system_message=_anonymize_system_message(trajectory.system_message, policy),
        turns=[_anonymize_turn(t, policy, salt) for t in trajectory.turns],
        final_reward=trajectory.final_reward,  # Preserve learning signal
    )


def _anonymize_metadata(
    metadata: OTSMetadata,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSMetadata:
    """Anonymize trajectory metadata."""
    return OTSMetadata(
        task_description=_summarize_text(metadata.task_description, max_len=100)
        if not policy.should_preserve_field("task_description")
        else metadata.task_description,
        domain=metadata.domain,
        timestamp_start=metadata.timestamp_start,
        timestamp_end=metadata.timestamp_end,
        duration_ms=metadata.duration_ms,
        agent_id=hash_identifier(metadata.agent_id, salt),
        framework=metadata.framework,
        environment=metadata.environment,
        outcome=metadata.outcome,
        feedback_score=metadata.feedback_score,
        human_reviewed=metadata.human_reviewed,
        tags=metadata.tags,
        parent_trajectory_id=hash_identifier(metadata.parent_trajectory_id, salt)
        if metadata.parent_trajectory_id
        else None,
    )


def _anonymize_context(
    context: OTSContext,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSContext:
    """Anonymize trajectory context."""
    return OTSContext(
        referrer=None,  # Always redact referrer
        user=_anonymize_user(context.user, policy, salt) if context.user else None,
        entities=[_anonymize_entity(e, policy, salt) for e in context.entities],
        resources=[_anonymize_resource(r, policy) for r in context.resources],
        custom_context=None,  # Always redact custom context
    )


def _anonymize_user(
    user: OTSUser,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSUser:
    """Anonymize user information."""
    return OTSUser(
        id=hash_identifier(user.id, salt),
        handle=None,  # Always redact handle
        org_id=hash_identifier(user.org_id, salt) if user.org_id else None,
        teams=None,  # Always redact team membership
        timezone=user.timezone,  # Timezone is safe to keep
    )


def _anonymize_entity(
    entity: OTSEntity,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSEntity:
    """Anonymize entity reference."""
    return OTSEntity(
        type=entity.type,  # Keep type
        id=hash_identifier(entity.id, salt),
        name=entity.name,  # Keep name (usually generic like "file", "api")
        metadata={},  # Redact all metadata
    )


def _anonymize_resource(
    resource: OTSResource,
    policy: AnonymizationPolicy,
) -> OTSResource:
    """Anonymize resource reference."""
    return OTSResource(
        type=resource.type,  # Keep type
        uri="[REDACTED]",  # Always redact URI
        accessed_at=resource.accessed_at,
    )


def _anonymize_system_message(
    system_message: Optional[OTSSystemMessage],
    policy: AnonymizationPolicy,
) -> Optional[OTSSystemMessage]:
    """Anonymize system message."""
    if not system_message:
        return None

    return OTSSystemMessage(
        content="[REDACTED]",  # Always redact system prompt
        timestamp=system_message.timestamp,
    )


def _anonymize_turn(
    turn: OTSTurn,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSTurn:
    """Anonymize a single turn."""
    return OTSTurn(
        turn_id=turn.turn_id,
        span_id=hash_identifier(turn.span_id, salt),
        parent_span_id=hash_identifier(turn.parent_span_id, salt)
        if turn.parent_span_id
        else None,
        timestamp=turn.timestamp,
        duration_ms=turn.duration_ms,
        error=turn.error,
        turn_reward=turn.turn_reward,
        messages=[_anonymize_message(m, policy, salt) for m in turn.messages],
        decisions=[_anonymize_decision(d, policy, salt) for d in turn.decisions],
    )


def _anonymize_message(
    msg: OTSMessage,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSMessage:
    """Anonymize message content."""
    return OTSMessage(
        message_id=hash_identifier(msg.message_id, salt),
        role=msg.role,
        timestamp=msg.timestamp,
        content=_anonymize_message_content(msg.content, policy),
        reasoning=None,  # Always redact reasoning
        visibility=msg.visibility,
        context_snapshot=_anonymize_context_snapshot(msg.context_snapshot, policy, salt)
        if msg.context_snapshot
        else None,
    )


def _anonymize_message_content(
    content: OTSMessageContent,
    policy: AnonymizationPolicy,
) -> OTSMessageContent:
    """Anonymize message content based on type."""
    if content.type.value == "tool_call" and content.data:
        # Preserve tool name, redact arguments
        return OTSMessageContent(
            type=content.type,
            text=None,
            data={
                "tool_call": {
                    "name": content.data.get("tool_call", {}).get("name", "[UNKNOWN]"),
                    "arguments": "[REDACTED]",
                }
            }
            if "tool_call" in content.data
            else {"type": "redacted"},
        )
    elif content.type.value == "tool_response":
        # Keep type indicator, redact actual response
        return OTSMessageContent(
            type=content.type,
            text=None,
            data={"type": "tool_response", "content": "[REDACTED]"},
        )
    else:
        # Redact all text content
        return OTSMessageContent(
            type=content.type,
            text="[REDACTED]" if content.text else None,
            data={"type": "redacted"} if content.data else None,
        )


def _anonymize_context_snapshot(
    snapshot: OTSContextSnapshot,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSContextSnapshot:
    """Anonymize context snapshot."""
    return OTSContextSnapshot(
        entities=[hash_identifier(e, salt) for e in snapshot.entities],
        available_tools=snapshot.available_tools
        if policy.should_preserve_field("available_tools")
        else [],
    )


def _anonymize_decision(
    decision: OTSDecision,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSDecision:
    """Anonymize decision while preserving action patterns."""
    return OTSDecision(
        decision_id=hash_identifier(decision.decision_id, salt),
        decision_type=decision.decision_type,
        state=_anonymize_decision_state(decision.state, policy)
        if decision.state
        else None,
        alternatives=_anonymize_alternatives(decision.alternatives, policy)
        if decision.alternatives
        else None,
        choice=_anonymize_choice(decision.choice, policy),
        consequence=_anonymize_consequence(decision.consequence, policy),
        evaluation=_anonymize_evaluation(decision.evaluation, policy, salt)
        if decision.evaluation
        else None,
        credit_assignment=decision.credit_assignment,  # Preserve learning signal
    )


def _anonymize_decision_state(
    state: OTSDecisionState,
    policy: AnonymizationPolicy,
) -> OTSDecisionState:
    """Anonymize decision state."""
    return OTSDecisionState(
        context_summary=_summarize_text(state.context_summary, max_len=50)
        if state.context_summary and policy.should_preserve_field("context_summary")
        else None,
        available_actions=state.available_actions
        if policy.should_preserve_field("available_actions")
        else [],
    )


def _anonymize_alternatives(
    alternatives: Dict[str, List[OTSAlternative]],
    policy: AnonymizationPolicy,
) -> Dict[str, List[OTSAlternative]]:
    """Anonymize decision alternatives."""
    if not policy.should_preserve_field("available_actions"):
        return {}

    result = {}
    for key, alts in alternatives.items():
        result[key] = [
            OTSAlternative(
                action=alt.action,
                rationale=_summarize_text(alt.rationale, max_len=50)
                if alt.rationale
                else None,
                rejected_reason=_summarize_text(alt.rejected_reason, max_len=50)
                if alt.rejected_reason
                else None,
            )
            for alt in alts
        ]
    return result


def _anonymize_choice(
    choice: OTSChoice,
    policy: AnonymizationPolicy,
) -> OTSChoice:
    """Anonymize the chosen action."""
    return OTSChoice(
        action=choice.action,  # Always preserve action name
        arguments=None,  # Always redact arguments
        rationale=_summarize_text(choice.rationale, max_len=100)
        if choice.rationale and policy.should_preserve_field("rationale")
        else None,
        confidence=choice.confidence,  # Preserve learning signal
    )


def _anonymize_consequence(
    consequence: OTSConsequence,
    policy: AnonymizationPolicy,
) -> OTSConsequence:
    """Anonymize decision consequence."""
    return OTSConsequence(
        success=consequence.success,  # Always preserve
        result_summary=_summarize_text(consequence.result_summary, max_len=50)
        if consequence.result_summary and policy.should_preserve_field("result_summary")
        else None,
        error_type=consequence.error_type,  # Always preserve for learning
    )


def _anonymize_evaluation(
    evaluation: OTSDecisionEvaluation,
    policy: AnonymizationPolicy,
    salt: str,
) -> OTSDecisionEvaluation:
    """Anonymize decision evaluation."""
    return OTSDecisionEvaluation(
        evaluator_id=hash_identifier(evaluation.evaluator_id, salt),
        score=evaluation.score,  # Preserve
        criteria_scores=evaluation.criteria_scores,  # Preserve
        feedback=None,  # Redact detailed feedback
        counterfactual=None,  # Redact counterfactual analysis
    )


def _summarize_text(text: Optional[str], max_len: int = 100) -> Optional[str]:
    """
    Summarize text to preserve intent without exposing details.

    Takes first sentence or truncates at max_len.
    """
    if not text:
        return None

    # Try to keep first sentence
    if "." in text[:max_len]:
        return text[: text.index(".") + 1]

    # Otherwise truncate
    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."
