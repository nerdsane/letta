"""
Pure OTS Analytics Functions.

Analytics that work directly on OTS trajectory data without
requiring any LLM enrichment. These provide immediate insights
from the raw trajectory structure.

Use cases:
- Dashboard visualizations
- Performance monitoring
- Error analysis
- Tool usage patterns
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ots.models import OTSTrajectory, DecisionType


@dataclass
class OTSAnalytics:
    """Container for all pure OTS analytics."""

    decision_success_rate: Dict[str, float]
    action_frequency: Dict[str, int]
    decision_type_breakdown: Dict[str, int]
    turn_distribution: Dict[int, int]
    error_type_frequency: Dict[str, int]
    trajectory_outcomes: Dict[str, int]
    total_trajectories: int
    total_turns: int
    total_decisions: int
    total_messages: int
    avg_turns_per_trajectory: float
    avg_decisions_per_turn: float
    overall_success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_success_rate": self.decision_success_rate,
            "action_frequency": self.action_frequency,
            "decision_type_breakdown": self.decision_type_breakdown,
            "turn_distribution": self.turn_distribution,
            "error_type_frequency": self.error_type_frequency,
            "trajectory_outcomes": self.trajectory_outcomes,
            "total_trajectories": self.total_trajectories,
            "total_turns": self.total_turns,
            "total_decisions": self.total_decisions,
            "total_messages": self.total_messages,
            "avg_turns_per_trajectory": self.avg_turns_per_trajectory,
            "avg_decisions_per_turn": self.avg_decisions_per_turn,
            "overall_success_rate": self.overall_success_rate,
        }


def compute_decision_success_rate(
    trajectories: List[OTSTrajectory],
) -> Dict[str, float]:
    """
    Compute success rate per action type.

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Dictionary mapping action name to success rate (0.0-1.0)
    """
    action_outcomes: Dict[str, List[bool]] = {}

    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                if decision.consequence and decision.consequence.success is not None:
                    action = decision.choice.action
                    if action not in action_outcomes:
                        action_outcomes[action] = []
                    action_outcomes[action].append(decision.consequence.success)

    return {
        action: sum(outcomes) / len(outcomes)
        for action, outcomes in action_outcomes.items()
        if outcomes
    }


def compute_action_frequency(
    trajectories: List[OTSTrajectory],
    top_n: Optional[int] = None,
) -> Dict[str, int]:
    """
    Count occurrences of each action across all trajectories.

    Args:
        trajectories: List of OTS trajectories
        top_n: If provided, return only top N actions

    Returns:
        Dictionary mapping action name to count, ordered by frequency
    """
    counter: Counter = Counter()

    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                counter[decision.choice.action] += 1

    if top_n:
        return dict(counter.most_common(top_n))
    return dict(counter.most_common())


def compute_decision_type_breakdown(
    trajectories: List[OTSTrajectory],
) -> Dict[str, int]:
    """
    Count decisions by type (tool_selection, parameter_choice, etc.).

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Dictionary mapping decision type to count
    """
    counter: Counter = Counter()

    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                dtype = (
                    decision.decision_type.value if decision.decision_type else "unknown"
                )
                counter[dtype] += 1

    return dict(counter)


def compute_turn_distribution(
    trajectories: List[OTSTrajectory],
) -> Dict[int, int]:
    """
    Distribution of turn counts across trajectories.

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Dictionary mapping turn count to number of trajectories with that count
    """
    counter: Counter = Counter()

    for traj in trajectories:
        counter[len(traj.turns)] += 1

    return dict(sorted(counter.items()))


def compute_error_type_frequency(
    trajectories: List[OTSTrajectory],
    top_n: Optional[int] = None,
) -> Dict[str, int]:
    """
    Count occurrences of each error type.

    Args:
        trajectories: List of OTS trajectories
        top_n: If provided, return only top N error types

    Returns:
        Dictionary mapping error type to count
    """
    counter: Counter = Counter()

    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                if decision.consequence and decision.consequence.error_type:
                    counter[decision.consequence.error_type] += 1

    if top_n:
        return dict(counter.most_common(top_n))
    return dict(counter.most_common())


def compute_trajectory_outcomes(
    trajectories: List[OTSTrajectory],
) -> Dict[str, int]:
    """
    Distribution of trajectory outcomes (from metadata).

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Dictionary mapping outcome type to count
    """
    counter: Counter = Counter()

    for traj in trajectories:
        if traj.metadata and traj.metadata.outcome:
            counter[traj.metadata.outcome.value] += 1
        else:
            counter["unknown"] += 1

    return dict(counter)


def compute_duration_stats(
    trajectories: List[OTSTrajectory],
) -> Dict[str, Optional[float]]:
    """
    Compute duration statistics across trajectories.

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Dictionary with min, max, avg, median durations in milliseconds
    """
    durations = [
        traj.metadata.duration_ms
        for traj in trajectories
        if traj.metadata and traj.metadata.duration_ms is not None
    ]

    if not durations:
        return {
            "min_ms": None,
            "max_ms": None,
            "avg_ms": None,
            "median_ms": None,
        }

    sorted_durations = sorted(durations)
    n = len(sorted_durations)

    return {
        "min_ms": min(durations),
        "max_ms": max(durations),
        "avg_ms": sum(durations) / n,
        "median_ms": sorted_durations[n // 2],
    }


def compute_confidence_distribution(
    trajectories: List[OTSTrajectory],
    buckets: int = 10,
) -> Dict[str, int]:
    """
    Distribution of decision confidence scores.

    Args:
        trajectories: List of OTS trajectories
        buckets: Number of buckets (default 10 for 0-10%, 10-20%, etc.)

    Returns:
        Dictionary mapping confidence bucket to count
    """
    bucket_size = 1.0 / buckets
    counter: Counter = Counter()

    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                if decision.choice.confidence is not None:
                    bucket_idx = min(
                        int(decision.choice.confidence / bucket_size), buckets - 1
                    )
                    bucket_label = f"{bucket_idx * 10}-{(bucket_idx + 1) * 10}%"
                    counter[bucket_label] += 1

    return dict(sorted(counter.items()))


def compute_pivotal_decisions(
    trajectories: List[OTSTrajectory],
) -> Dict[str, Any]:
    """
    Analyze pivotal decisions across trajectories.

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Dictionary with pivotal decision stats
    """
    pivotal_by_action: Counter = Counter()
    total_pivotal = 0
    total_decisions = 0

    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                total_decisions += 1
                if decision.credit_assignment and decision.credit_assignment.pivotal:
                    total_pivotal += 1
                    pivotal_by_action[decision.choice.action] += 1

    return {
        "total_pivotal": total_pivotal,
        "pivotal_rate": total_pivotal / total_decisions if total_decisions else 0.0,
        "pivotal_by_action": dict(pivotal_by_action.most_common()),
    }


def get_pure_ots_analytics(
    trajectories: List[OTSTrajectory],
) -> OTSAnalytics:
    """
    Compute all pure OTS analytics in one call.

    This is the main entry point for getting a complete analytics
    snapshot from a list of trajectories.

    Args:
        trajectories: List of OTS trajectories

    Returns:
        OTSAnalytics dataclass with all computed metrics
    """
    # Count totals
    total_turns = sum(len(traj.turns) for traj in trajectories)
    total_decisions = sum(
        len(decision)
        for traj in trajectories
        for turn in traj.turns
        for decision in [turn.decisions]
    )
    total_messages = sum(
        len(turn.messages) for traj in trajectories for turn in traj.turns
    )

    # Compute overall success rate
    success_count = 0
    outcome_count = 0
    for traj in trajectories:
        for turn in traj.turns:
            for decision in turn.decisions:
                if decision.consequence and decision.consequence.success is not None:
                    outcome_count += 1
                    if decision.consequence.success:
                        success_count += 1

    overall_success_rate = success_count / outcome_count if outcome_count else 0.0

    return OTSAnalytics(
        decision_success_rate=compute_decision_success_rate(trajectories),
        action_frequency=compute_action_frequency(trajectories),
        decision_type_breakdown=compute_decision_type_breakdown(trajectories),
        turn_distribution=compute_turn_distribution(trajectories),
        error_type_frequency=compute_error_type_frequency(trajectories),
        trajectory_outcomes=compute_trajectory_outcomes(trajectories),
        total_trajectories=len(trajectories),
        total_turns=total_turns,
        total_decisions=total_decisions,
        total_messages=total_messages,
        avg_turns_per_trajectory=total_turns / len(trajectories)
        if trajectories
        else 0.0,
        avg_decisions_per_turn=total_decisions / total_turns if total_turns else 0.0,
        overall_success_rate=overall_success_rate,
    )


def get_analytics_summary(trajectories: List[OTSTrajectory]) -> str:
    """
    Generate a human-readable analytics summary.

    Args:
        trajectories: List of OTS trajectories

    Returns:
        Formatted string summary
    """
    analytics = get_pure_ots_analytics(trajectories)

    lines = [
        f"=== OTS Analytics Summary ===",
        f"",
        f"Totals:",
        f"  Trajectories: {analytics.total_trajectories}",
        f"  Turns: {analytics.total_turns}",
        f"  Decisions: {analytics.total_decisions}",
        f"  Messages: {analytics.total_messages}",
        f"",
        f"Averages:",
        f"  Turns per trajectory: {analytics.avg_turns_per_trajectory:.1f}",
        f"  Decisions per turn: {analytics.avg_decisions_per_turn:.1f}",
        f"  Overall success rate: {analytics.overall_success_rate:.1%}",
        f"",
        f"Top Actions:",
    ]

    for action, count in list(analytics.action_frequency.items())[:5]:
        rate = analytics.decision_success_rate.get(action, 0)
        lines.append(f"  {action}: {count} calls ({rate:.0%} success)")

    lines.append("")
    lines.append("Outcomes:")
    for outcome, count in analytics.trajectory_outcomes.items():
        lines.append(f"  {outcome}: {count}")

    if analytics.error_type_frequency:
        lines.append("")
        lines.append("Top Errors:")
        for error, count in list(analytics.error_type_frequency.items())[:3]:
            lines.append(f"  {error}: {count}")

    return "\n".join(lines)
