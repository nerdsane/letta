"""
OTS Analytics Module.

Provides analytics functions that work on pure OTS data
without requiring LLM enrichment.

These functions compute metrics directly from the OTS structure:
- Decision success rates
- Action frequency distributions
- Turn distributions
- Error type breakdowns
"""

from ots.analytics.pure_ots import (
    compute_action_frequency,
    compute_decision_success_rate,
    compute_decision_type_breakdown,
    compute_error_type_frequency,
    compute_trajectory_outcomes,
    compute_turn_distribution,
    get_pure_ots_analytics,
    OTSAnalytics,
)

__all__ = [
    # Individual functions
    "compute_decision_success_rate",
    "compute_action_frequency",
    "compute_decision_type_breakdown",
    "compute_turn_distribution",
    "compute_error_type_frequency",
    "compute_trajectory_outcomes",
    # Aggregate
    "get_pure_ots_analytics",
    "OTSAnalytics",
]
