"""
Tests for trajectory converter service.

The trajectory converter transforms completed agent runs (Run + Steps + Messages)
into trajectory format for continual learning and retrieval.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from letta.orm.run import Run
from letta.orm.step import Step
from letta.orm.message import Message
from letta.schemas.enums import MessageRole
from letta.services.trajectory_converter import TrajectoryConverter


@pytest.fixture
def converter():
    """Create a TrajectoryConverter instance for testing."""
    return TrajectoryConverter()


@pytest.fixture
def mock_run():
    """Create a mock completed Run."""
    run = Mock(spec=Run)
    run.id = "run-123"
    run.agent_id = "agent-456"
    run.status = "completed"
    run.stop_reason = "end_turn"
    run.created_at = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    run.completed_at = datetime(2025, 1, 1, 10, 5, 30, tzinfo=timezone.utc)
    run.metadata_ = {"run_type": "send_message_streaming"}
    return run


@pytest.fixture
def mock_steps():
    """Create mock Steps for a run."""
    step1 = Mock(spec=Step)
    step1.id = "step-1"
    step1.model = "gpt-4"
    step1.prompt_tokens = 100
    step1.completion_tokens = 50
    step1.stop_reason = "end_turn"

    step2 = Mock(spec=Step)
    step2.id = "step-2"
    step2.model = "gpt-4"
    step2.prompt_tokens = 150
    step2.completion_tokens = 75
    step2.stop_reason = "tool_calls"

    return [step1, step2]


@pytest.fixture
def mock_messages():
    """Create mock Messages for a run."""
    # User message
    user_msg = Mock(spec=Message)
    user_msg.id = "msg-1"
    user_msg.role = MessageRole.user
    user_msg.step_id = "step-1"
    user_msg.created_at = datetime(2025, 1, 1, 10, 0, 10, tzinfo=timezone.utc)
    user_msg.text = "Hello, tell me a story"
    user_msg.content = None
    user_msg.tool_calls = None
    user_msg.tool_call_id = None

    # Assistant message
    assistant_msg = Mock(spec=Message)
    assistant_msg.id = "msg-2"
    assistant_msg.role = MessageRole.assistant
    assistant_msg.step_id = "step-1"
    assistant_msg.created_at = datetime(2025, 1, 1, 10, 0, 15, tzinfo=timezone.utc)
    assistant_msg.text = None
    assistant_msg.content = [
        Mock(type="text", text="Once upon a time...")
    ]
    assistant_msg.tool_calls = None
    assistant_msg.tool_call_id = None

    # Tool call message
    tool_call_msg = Mock(spec=Message)
    tool_call_msg.id = "msg-3"
    tool_call_msg.role = MessageRole.assistant
    tool_call_msg.step_id = "step-2"
    tool_call_msg.created_at = datetime(2025, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
    tool_call_msg.text = None
    tool_call_msg.content = None
    tool_call_msg.tool_call_id = None
    tool_call_function = Mock()
    tool_call_function.name = "search_memory"
    tool_call_function.arguments = '{"query": "stories"}'

    tool_call_msg.tool_calls = [
        Mock(
            id="call-1",
            type="function",
            function=tool_call_function
        )
    ]

    return [user_msg, assistant_msg, tool_call_msg]


@pytest.mark.asyncio
async def test_from_run_basic(converter, mock_run, mock_steps, mock_messages):
    """Test basic conversion from Run to trajectory."""
    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)

    assert trajectory.agent_id == "agent-456"
    assert trajectory.data["run_id"] == "run-123"
    assert "metadata" in trajectory.data
    assert "turns" in trajectory.data
    assert "outcome" in trajectory.data


@pytest.mark.asyncio
async def test_extract_metadata(converter, mock_run, mock_steps, mock_messages):
    """Test metadata extraction from run."""
    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    metadata = trajectory.data["metadata"]

    # Check timing
    assert metadata["start_time"] == "2025-01-01T10:00:00+00:00"
    assert metadata["end_time"] == "2025-01-01T10:05:30+00:00"
    assert metadata["duration_ns"] == 330_000_000_000  # 5.5 minutes in nanoseconds

    # Check status
    assert metadata["status"] == "completed"
    assert metadata["stop_reason"] == "end_turn"

    # Check counts
    assert metadata["step_count"] == 2
    assert metadata["message_count"] == 3

    # Check tokens
    assert metadata["input_tokens"] == 250  # 100 + 150
    assert metadata["output_tokens"] == 125  # 50 + 75
    assert metadata["total_tokens"] == 375

    # Check models
    assert metadata["models"] == ["gpt-4"]

    # Check tools
    assert "search_memory" in metadata["tools_used"]


@pytest.mark.asyncio
async def test_structure_turns(converter, mock_run, mock_steps, mock_messages):
    """Test turn structuring from steps and messages."""
    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    turns = trajectory.data["turns"]

    assert len(turns) == 2

    # First turn
    turn1 = turns[0]
    assert turn1["step_id"] == "step-1"
    assert turn1["model"] == "gpt-4"
    assert turn1["input_tokens"] == 100
    assert turn1["output_tokens"] == 50
    assert len(turn1["messages"]) == 2  # user + assistant

    # Second turn
    turn2 = turns[1]
    assert turn2["step_id"] == "step-2"
    assert len(turn2["messages"]) == 1  # tool call message


@pytest.mark.asyncio
async def test_message_to_dict_user_message(converter, mock_run, mock_steps, mock_messages):
    """Test user message conversion."""
    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    turns = trajectory.data["turns"]

    user_message = turns[0]["messages"][0]
    assert user_message["message_id"] == "msg-1"
    assert user_message["role"] == MessageRole.user
    assert user_message["content"][0]["text"] == "Hello, tell me a story"


@pytest.mark.asyncio
async def test_message_to_dict_with_tool_calls(converter, mock_run, mock_steps, mock_messages):
    """Test message with tool calls conversion."""
    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    turns = trajectory.data["turns"]

    tool_message = turns[1]["messages"][0]
    assert "tool_calls" in tool_message
    assert len(tool_message["tool_calls"]) == 1
    assert tool_message["tool_calls"][0]["function"]["name"] == "search_memory"


@pytest.mark.asyncio
async def test_determine_outcome_success(converter, mock_run, mock_steps, mock_messages):
    """Test outcome determination for successful run."""
    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    outcome = trajectory.data["outcome"]

    assert outcome["type"] == "success"
    assert outcome["confidence"] >= 0.7
    assert "Run completed successfully" in outcome["reasoning"]


@pytest.mark.asyncio
async def test_determine_outcome_failure(converter, mock_run, mock_steps, mock_messages):
    """Test outcome determination for failed run."""
    mock_run.status = "failed"
    mock_run.metadata_ = {"error": "Connection timeout"}

    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    outcome = trajectory.data["outcome"]

    assert outcome["type"] == "failure"
    assert outcome["confidence"] >= 0.8
    assert "Run failed with error" in outcome["reasoning"]


@pytest.mark.asyncio
async def test_determine_outcome_partial_success(converter, mock_run, mock_steps, mock_messages):
    """Test outcome determination for partial success."""
    mock_run.stop_reason = "max_tokens"

    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    outcome = trajectory.data["outcome"]

    assert outcome["type"] == "partial_success"
    assert any("Hit token limit" in r for r in outcome["reasoning"])


@pytest.mark.asyncio
async def test_determine_outcome_cancelled(converter, mock_run, mock_steps, mock_messages):
    """Test outcome determination for cancelled run."""
    mock_run.status = "cancelled"

    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    outcome = trajectory.data["outcome"]

    assert outcome["type"] == "failure"
    assert any("cancelled" in r.lower() for r in outcome["reasoning"])


@pytest.mark.asyncio
async def test_metadata_with_no_tools(converter, mock_run, mock_steps):
    """Test metadata extraction when no tools are used."""
    # Messages with no tool calls
    simple_messages = [
        Mock(
            id="msg-1",
            role=MessageRole.user,
            step_id="step-1",
            created_at=datetime.now(timezone.utc),
            text="Hello",
            content=None,
            tool_calls=None,
            tool_call_id=None
        )
    ]

    trajectory = await converter.from_run(mock_run, mock_steps, simple_messages)
    metadata = trajectory.data["metadata"]

    assert metadata["tools_used"] == []


@pytest.mark.asyncio
async def test_metadata_with_null_timestamps(converter, mock_steps, mock_messages):
    """Test metadata extraction with null timestamps."""
    mock_run = Mock(spec=Run)
    mock_run.id = "run-123"
    mock_run.agent_id = "agent-456"
    mock_run.status = "completed"
    mock_run.stop_reason = "end_turn"
    mock_run.created_at = None
    mock_run.completed_at = None
    mock_run.metadata_ = None

    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    metadata = trajectory.data["metadata"]

    assert metadata["start_time"] is None
    assert metadata["end_time"] is None
    assert metadata["duration_ns"] is None


@pytest.mark.asyncio
async def test_turns_with_no_messages_for_step(converter, mock_run, mock_steps):
    """Test turn structuring when a step has no messages."""
    # Create messages only for first step
    messages_for_step_1 = [
        Mock(
            id="msg-1",
            role=MessageRole.user,
            step_id="step-1",
            created_at=datetime.now(timezone.utc),
            text="Hello",
            content=None,
            tool_calls=None,
            tool_call_id=None
        )
    ]

    trajectory = await converter.from_run(mock_run, mock_steps, messages_for_step_1)
    turns = trajectory.data["turns"]

    assert len(turns) == 2
    assert len(turns[0]["messages"]) == 1
    assert len(turns[1]["messages"]) == 0  # Step 2 has no messages


@pytest.mark.asyncio
async def test_timezone_aware_duration_calculation(converter, mock_steps, mock_messages):
    """Test duration calculation with timezone-aware timestamps."""
    mock_run = Mock(spec=Run)
    mock_run.id = "run-123"
    mock_run.agent_id = "agent-456"
    mock_run.status = "completed"
    mock_run.stop_reason = "end_turn"
    # Mixed timezone awareness
    mock_run.created_at = datetime(2025, 1, 1, 10, 0, 0)  # naive
    mock_run.completed_at = datetime(2025, 1, 1, 10, 1, 0, tzinfo=timezone.utc)  # aware
    mock_run.metadata_ = {}

    trajectory = await converter.from_run(mock_run, mock_steps, mock_messages)
    metadata = trajectory.data["metadata"]

    # Should not raise error and should calculate duration
    assert metadata["duration_ns"] is not None
    assert metadata["duration_ns"] > 0


@pytest.mark.asyncio
async def test_message_content_structured_format(converter, mock_run, mock_steps):
    """Test message conversion with new structured content format."""
    message_with_content = Mock(spec=Message)
    message_with_content.id = "msg-1"
    message_with_content.role = MessageRole.assistant
    message_with_content.step_id = "step-1"
    message_with_content.created_at = datetime.now(timezone.utc)
    message_with_content.text = None
    message_with_content.content = [
        Mock(type="text", text="First part"),
        Mock(type="reasoning", text="Internal thought")
    ]
    message_with_content.tool_calls = None
    message_with_content.tool_call_id = None

    trajectory = await converter.from_run(mock_run, mock_steps, [message_with_content])
    turns = trajectory.data["turns"]

    msg = turns[0]["messages"][0]
    assert len(msg["content"]) == 2
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][1]["type"] == "reasoning"
