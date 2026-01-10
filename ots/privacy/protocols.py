"""
Anonymization policy protocols and implementations.

Policies control how trajectories are anonymized:
- Which fields to hash (IDs)
- Which fields to redact (content)
- How to transform specific values
"""

from typing import Any, Protocol, Set


class AnonymizationPolicy(Protocol):
    """Protocol for customizable anonymization strategies."""

    def should_hash_id(self, field_name: str) -> bool:
        """Whether to hash this ID field."""
        ...

    def should_redact_content(self, field_name: str) -> bool:
        """Whether to redact this content field."""
        ...

    def transform_value(self, field_name: str, value: Any) -> Any:
        """Custom transformation for specific fields."""
        ...

    def should_preserve_field(self, field_name: str) -> bool:
        """Whether to preserve this field for learning value."""
        ...


class DefaultAnonymizationPolicy:
    """
    Default policy that hashes IDs and redacts message content.

    Suitable for general-purpose sharing where privacy is paramount.
    """

    ID_FIELDS: Set[str] = {
        "trajectory_id",
        "user_id",
        "agent_id",
        "organization_id",
        "org_id",
        "id",
        "handle",
        "message_id",
        "decision_id",
        "span_id",
        "parent_span_id",
        "evaluator_id",
    }

    CONTENT_FIELDS: Set[str] = {
        "content",
        "text",
        "data",
        "arguments",
        "result",
        "result_summary",
        "context_summary",
        "custom_context",
        "feedback",
        "explanation",
        "uri",
        "referrer",
    }

    PRESERVED_FIELDS: Set[str] = {
        # Structure
        "turn_id",
        "decision_type",
        "role",
        "type",
        "version",
        # Learning signal
        "success",
        "error",
        "error_type",
        "outcome",
        "score",
        "confidence",
        "contribution_to_outcome",
        "pivotal",
        "final_reward",
        "turn_reward",
        "feedback_score",
        # Timing
        "timestamp",
        "timestamp_start",
        "timestamp_end",
        "duration_ms",
        "accessed_at",
        # Metadata
        "domain",
        "framework",
        "environment",
        "tags",
        "action",
        "name",
    }

    def should_hash_id(self, field_name: str) -> bool:
        """Hash all ID fields."""
        return field_name in self.ID_FIELDS or field_name.endswith("_id")

    def should_redact_content(self, field_name: str) -> bool:
        """Redact content fields."""
        return field_name in self.CONTENT_FIELDS

    def should_preserve_field(self, field_name: str) -> bool:
        """Preserve fields with learning value."""
        return field_name in self.PRESERVED_FIELDS

    def transform_value(self, field_name: str, value: Any) -> Any:
        """No custom transforms by default."""
        return value


class LearningPreservingPolicy(DefaultAnonymizationPolicy):
    """
    Policy optimized for preserving learning signal.

    Less aggressive redaction - keeps rationale patterns,
    action names, and outcome details for training value.
    """

    # Keep more content for learning
    CONTENT_FIELDS: Set[str] = {
        "text",
        "data",
        "arguments",
        "custom_context",
        "uri",
        "referrer",
    }

    PRESERVED_FIELDS: Set[str] = DefaultAnonymizationPolicy.PRESERVED_FIELDS | {
        # Keep rationale patterns (but summarize)
        "rationale",
        "rejected_reason",
        "task_description",
        "result_summary",
        "context_summary",
        # Keep available actions for decision context
        "available_actions",
        "available_tools",
    }

    def should_redact_content(self, field_name: str) -> bool:
        """Less aggressive redaction for learning."""
        return field_name in self.CONTENT_FIELDS
