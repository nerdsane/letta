"""
Tests for the evaluation tools and executor.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta.constants import EVALUATION_TOOLS
from letta.schemas.agent import AgentState
from letta.schemas.enums import ToolType
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.services.tool_executor.evaluation_tool_executor import LettaEvaluationToolExecutor
from letta.services.tool_executor.tool_execution_manager import ToolExecutorFactory


def test_evaluation_tools_defined():
    """Test that evaluation tools are defined in constants."""
    assert "assess_output_quality" in EVALUATION_TOOLS
    assert "check_logical_consistency" in EVALUATION_TOOLS
    assert "compare_versions" in EVALUATION_TOOLS
    assert "analyze_information_gain" in EVALUATION_TOOLS
    assert len(EVALUATION_TOOLS) == 4


def test_evaluation_tool_type_exists():
    """Test that LETTA_EVALUATION ToolType exists."""
    assert hasattr(ToolType, "LETTA_EVALUATION")
    assert ToolType.LETTA_EVALUATION == "letta_evaluation"


def test_evaluation_executor_registered():
    """Test that LettaEvaluationToolExecutor is registered in factory."""
    # Mock the manager dependencies
    message_manager = MagicMock()
    agent_manager = MagicMock()
    block_manager = MagicMock()
    run_manager = MagicMock()
    passage_manager = MagicMock()
    actor = MagicMock()

    executor = ToolExecutorFactory.get_executor(
        tool_type=ToolType.LETTA_EVALUATION,
        message_manager=message_manager,
        agent_manager=agent_manager,
        block_manager=block_manager,
        run_manager=run_manager,
        passage_manager=passage_manager,
        actor=actor,
    )

    assert isinstance(executor, LettaEvaluationToolExecutor)


@pytest.mark.asyncio
async def test_assess_output_quality_executor():
    """Test the assess_output_quality tool executor."""
    # Create mock dependencies
    message_manager = MagicMock()
    agent_manager = MagicMock()
    block_manager = MagicMock()
    run_manager = MagicMock()
    passage_manager = MagicMock()
    actor = MagicMock(spec=User)
    actor.organization_id = "test-org"

    # Create agent state mock
    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "test-agent-id"

    # Create tool mock
    tool = MagicMock(spec=Tool)
    tool.name = "assess_output_quality"
    tool.tool_type = ToolType.LETTA_EVALUATION

    # Create executor
    executor = LettaEvaluationToolExecutor(
        message_manager=message_manager,
        agent_manager=agent_manager,
        block_manager=block_manager,
        run_manager=run_manager,
        passage_manager=passage_manager,
        actor=actor,
    )

    # Mock the LLM call to return a valid response
    mock_llm_response = json.dumps(
        {
            "score": 0.85,
            "reasoning": "The content is clear and well-structured.",
            "strengths": ["Clear language", "Good organization"],
            "improvements": ["Could add more examples"],
            "meets_criteria": True,
        }
    )

    with patch.object(executor, "_call_llm_for_evaluation", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Execute the tool
        result = await executor.execute(
            function_name="assess_output_quality",
            function_args={"content": "Test content", "rubric": "Clear and concise", "content_type": "text"},
            tool=tool,
            actor=actor,
            agent_state=agent_state,
        )

        # Verify the result
        assert result.status == "success"
        assert result.func_return is not None

        # Parse the returned JSON
        response_data = json.loads(result.func_return)
        assert "score" in response_data
        assert "reasoning" in response_data
        assert "strengths" in response_data
        assert "improvements" in response_data
        assert "meets_criteria" in response_data


@pytest.mark.asyncio
async def test_check_logical_consistency_executor():
    """Test the check_logical_consistency tool executor."""
    # Create mock dependencies
    message_manager = MagicMock()
    agent_manager = MagicMock()
    block_manager = MagicMock()
    run_manager = MagicMock()
    passage_manager = MagicMock()
    actor = MagicMock(spec=User)
    actor.organization_id = "test-org"

    # Create agent state mock
    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "test-agent-id"

    # Create tool mock
    tool = MagicMock(spec=Tool)
    tool.name = "check_logical_consistency"
    tool.tool_type = ToolType.LETTA_EVALUATION

    # Create executor
    executor = LettaEvaluationToolExecutor(
        message_manager=message_manager,
        agent_manager=agent_manager,
        block_manager=block_manager,
        run_manager=run_manager,
        passage_manager=passage_manager,
        actor=actor,
    )

    # Mock the LLM call to return a valid response
    mock_llm_response = json.dumps({"consistent": True, "contradictions": [], "checks_performed": 5})

    with patch.object(executor, "_call_llm_for_evaluation", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Execute the tool
        result = await executor.execute(
            function_name="check_logical_consistency",
            function_args={"content": "Test content", "rules": ["Rule 1", "Rule 2"], "format": "text"},
            tool=tool,
            actor=actor,
            agent_state=agent_state,
        )

        # Verify the result
        assert result.status == "success"
        assert result.func_return is not None

        # Parse the returned JSON
        response_data = json.loads(result.func_return)
        assert "consistent" in response_data
        assert "contradictions" in response_data
        assert "checks_performed" in response_data


@pytest.mark.asyncio
async def test_compare_versions_executor():
    """Test the compare_versions tool executor."""
    # Create mock dependencies
    message_manager = MagicMock()
    agent_manager = MagicMock()
    block_manager = MagicMock()
    run_manager = MagicMock()
    passage_manager = MagicMock()
    actor = MagicMock(spec=User)
    actor.organization_id = "test-org"

    # Create agent state mock
    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "test-agent-id"

    # Create tool mock
    tool = MagicMock(spec=Tool)
    tool.name = "compare_versions"
    tool.tool_type = ToolType.LETTA_EVALUATION

    # Create executor
    executor = LettaEvaluationToolExecutor(
        message_manager=message_manager,
        agent_manager=agent_manager,
        block_manager=block_manager,
        run_manager=run_manager,
        passage_manager=passage_manager,
        actor=actor,
    )

    # Mock the LLM call to return a valid response
    mock_llm_response = json.dumps(
        {
            "improved": True,
            "changes": ["Added more detail"],
            "better_aspects": ["More comprehensive"],
            "worse_aspects": [],
            "recommendation": "keep_current",
        }
    )

    with patch.object(executor, "_call_llm_for_evaluation", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Execute the tool
        result = await executor.execute(
            function_name="compare_versions",
            function_args={"current": "Version 2", "previous": "Version 1", "comparison_criteria": "quality"},
            tool=tool,
            actor=actor,
            agent_state=agent_state,
        )

        # Verify the result
        assert result.status == "success"
        assert result.func_return is not None

        # Parse the returned JSON
        response_data = json.loads(result.func_return)
        assert "improved" in response_data
        assert "changes" in response_data
        assert "better_aspects" in response_data
        assert "worse_aspects" in response_data
        assert "recommendation" in response_data


@pytest.mark.asyncio
async def test_analyze_information_gain_executor():
    """Test the analyze_information_gain tool executor."""
    # Create mock dependencies
    message_manager = MagicMock()
    agent_manager = MagicMock()
    block_manager = MagicMock()
    run_manager = MagicMock()
    passage_manager = MagicMock()
    actor = MagicMock(spec=User)
    actor.organization_id = "test-org"

    # Create agent state mock
    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "test-agent-id"

    # Create tool mock
    tool = MagicMock(spec=Tool)
    tool.name = "analyze_information_gain"
    tool.tool_type = ToolType.LETTA_EVALUATION

    # Create executor
    executor = LettaEvaluationToolExecutor(
        message_manager=message_manager,
        agent_manager=agent_manager,
        block_manager=block_manager,
        run_manager=run_manager,
        passage_manager=passage_manager,
        actor=actor,
    )

    # Mock the LLM call to return a valid response
    mock_llm_response = json.dumps(
        {
            "information_gain": 0.7,
            "new_facts": ["New fact 1", "New fact 2"],
            "insights": ["Insight 1"],
            "significance": "moderate",
        }
    )

    with patch.object(executor, "_call_llm_for_evaluation", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Execute the tool
        result = await executor.execute(
            function_name="analyze_information_gain",
            function_args={"after": "After state", "before": "Before state", "metric": "novelty"},
            tool=tool,
            actor=actor,
            agent_state=agent_state,
        )

        # Verify the result
        assert result.status == "success"
        assert result.func_return is not None

        # Parse the returned JSON
        response_data = json.loads(result.func_return)
        assert "information_gain" in response_data
        assert "new_facts" in response_data
        assert "insights" in response_data
        assert "significance" in response_data


@pytest.mark.asyncio
async def test_executor_error_handling():
    """Test that the executor handles errors gracefully."""
    # Create mock dependencies
    message_manager = MagicMock()
    agent_manager = MagicMock()
    block_manager = MagicMock()
    run_manager = MagicMock()
    passage_manager = MagicMock()
    actor = MagicMock(spec=User)
    actor.organization_id = "test-org"

    # Create agent state mock
    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "test-agent-id"

    # Create tool mock
    tool = MagicMock(spec=Tool)
    tool.name = "assess_output_quality"
    tool.tool_type = ToolType.LETTA_EVALUATION

    # Create executor
    executor = LettaEvaluationToolExecutor(
        message_manager=message_manager,
        agent_manager=agent_manager,
        block_manager=block_manager,
        run_manager=run_manager,
        passage_manager=passage_manager,
        actor=actor,
    )

    # Mock the LLM call to raise an exception
    with patch.object(executor, "_call_llm_for_evaluation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("LLM API error")

        # Execute the tool
        result = await executor.execute(
            function_name="assess_output_quality",
            function_args={"content": "Test content", "rubric": "Clear and concise", "content_type": "text"},
            tool=tool,
            actor=actor,
            agent_state=agent_state,
        )

        # Verify the error is handled
        assert result.status == "error"
        assert result.stderr is not None
        assert len(result.stderr) > 0
