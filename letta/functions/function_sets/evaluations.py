"""
Evaluation tools for agent self-assessment.

These tools enable agents to evaluate the quality of their work:
- Quality assessment against custom rubrics
- Logical consistency checking
- Version comparison for iterative improvement
- Information gain analysis

Design Philosophy (Bitter Lesson):
- Tools not workflows - Agent chooses when/if to use
- Parameters not hard-coding - Flexible via inputs (rubrics, criteria)
- Feedback not commands - Return information, agent decides action
- Scales with better models - Better judgment, not more rules
"""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState


def assess_output_quality(
    agent_state: "AgentState",
    content: str,
    rubric: str,
    content_type: str = "text",
) -> str:
    """
    Evaluate quality of content against a custom rubric using LLM-as-judge.

    Generic tool for ANY agent to self-assess their work. Agent provides
    content and quality criteria (rubric). Returns structured feedback.

    Use this when you want to check if your output meets specific quality
    criteria. The rubric should describe what "good" looks like.

    Args:
        content: The output to evaluate (text, JSON, code, etc.)
        rubric: Quality criteria as natural language description
        content_type: Type hint for parsing (text, json, python, markdown)

    Returns:
        JSON string with evaluation results:
        {
            "score": float (0.0-1.0),
            "reasoning": str,
            "strengths": [str],
            "improvements": [str],
            "meets_criteria": bool
        }

    Examples:
        # Writing assistant checks draft
        assess_output_quality(
            agent_state,
            content=draft,
            rubric="Clear, concise, engaging. No jargon. Active voice preferred.",
            content_type="text"
        )

        # Code agent checks implementation
        assess_output_quality(
            agent_state,
            content=code,
            rubric="Clean, readable, follows PEP8. Has docstrings. Handles errors.",
            content_type="python"
        )

        # World-building agent checks consistency
        assess_output_quality(
            agent_state,
            content=json.dumps(world),
            rubric="Logically consistent. Abstract roles (not names). Deep research evident.",
            content_type="json"
        )
    """
    raise NotImplementedError("This tool is executed by EvaluationToolExecutor. Contact Letta if you see this error.")


def check_logical_consistency(
    agent_state: "AgentState",
    content: str,
    rules: Optional[List[str]] = None,
    format: str = "text",
) -> str:
    """
    Check content for logical contradictions.

    Generic tool for ANY agent working with structured information.
    Can provide explicit rules to check against, or let the tool find
    implicit contradictions.

    Use this when you want to verify that your content doesn't contain
    logical conflicts or contradictions.

    Args:
        content: Content to check (text, JSON, etc.)
        rules: Optional explicit rules to verify against
        format: Format hint (text, json, yaml)

    Returns:
        JSON string with consistency check results:
        {
            "consistent": bool,
            "contradictions": [
                {
                    "elements": [str],  # What contradicts
                    "description": str,  # Why it contradicts
                    "severity": "minor" | "major"
                }
            ],
            "checks_performed": int
        }

    Examples:
        # Contract review agent
        check_logical_consistency(
            agent_state,
            content=contract_text,
            rules=[
                "Payment terms must be <= 90 days",
                "No conflicting clauses about termination"
            ],
            format="text"
        )

        # Data validation agent
        check_logical_consistency(
            agent_state,
            content=json.dumps(schema),
            rules=[
                "All foreign keys must reference existing tables",
                "No circular dependencies"
            ],
            format="json"
        )

        # World-building agent
        check_logical_consistency(
            agent_state,
            content=json.dumps(world),
            format="json"
        )
    """
    raise NotImplementedError("This tool is executed by EvaluationToolExecutor. Contact Letta if you see this error.")


def compare_versions(
    agent_state: "AgentState",
    current: str,
    previous: str,
    comparison_criteria: str = "quality, novelty, accuracy",
) -> str:
    """
    Compare two versions of content to measure improvement.

    Generic tool for iterative refinement. Agent can assess whether
    their revision improved or regressed on specific criteria.

    Use this when you've made changes and want to know if they're
    actually improvements.

    Args:
        current: Current version of content
        previous: Previous version to compare against
        comparison_criteria: What to compare (comma-separated)

    Returns:
        JSON string with comparison results:
        {
            "improved": bool,
            "changes": [str],  # What changed
            "better_aspects": [str],  # What improved
            "worse_aspects": [str],  # What regressed
            "recommendation": "keep_current" | "revert" | "iterate_more"
        }

    Examples:
        # Writing agent compares drafts
        compare_versions(
            agent_state,
            current=revised_draft,
            previous=first_draft,
            comparison_criteria="clarity, completeness, tone"
        )

        # Code agent compares implementations
        compare_versions(
            agent_state,
            current=refactored_code,
            previous=original_code,
            comparison_criteria="readability, performance, test_coverage"
        )

        # World-building agent compares iterations
        compare_versions(
            agent_state,
            current=json.dumps(world_v2),
            previous=json.dumps(world_v1),
            comparison_criteria="consistency, depth, abstraction"
        )
    """
    raise NotImplementedError("This tool is executed by EvaluationToolExecutor. Contact Letta if you see this error.")


def analyze_information_gain(
    agent_state: "AgentState",
    after: str,
    before: str,
    metric: str = "novelty",
) -> str:
    """
    Measure what new information was added between versions.

    Helps agents decide: "Did I learn something new?" or "Did I add
    meaningful depth?" Focus is on novelty and significance of changes,
    not just differences.

    Use this when you want to assess whether your work added genuinely
    new insights or just surface-level changes.

    Args:
        after: Later state with (potentially) more information
        before: Earlier state to compare against
        metric: What to measure (novelty, depth, breadth, significance)

    Returns:
        JSON string with information gain analysis:
        {
            "information_gain": float (0.0-1.0),
            "new_facts": [str],
            "insights": [str],
            "significance": "trivial" | "minor" | "moderate" | "major" | "breakthrough"
        }

    Examples:
        # Research agent measures learning
        analyze_information_gain(
            agent_state,
            after=research_output,
            before=initial_knowledge,
            metric="depth"
        )

        # World-building agent measures depth added
        analyze_information_gain(
            agent_state,
            after=json.dumps(world_v2),
            before=json.dumps(world_v1),
            metric="novelty"
        )

        # Data analysis agent measures insights
        analyze_information_gain(
            agent_state,
            after=analysis_results,
            before=raw_data_summary,
            metric="significance"
        )
    """
    raise NotImplementedError("This tool is executed by EvaluationToolExecutor. Contact Letta if you see this error.")
