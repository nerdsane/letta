# Trajectories: Agent Execution Capture for Continual Learning

Trajectories capture complete execution traces of agent runs, enabling continual learning through retrieval and analysis of past experiences.

## Overview

A **trajectory** is a structured record of what an agent DID during execution - the decisions made, tools used, reasoning performed, and outcomes achieved. Unlike logs which focus on debugging, trajectories are optimized for learning: they can be searched semantically, scored for quality, and retrieved as examples for future runs.

### Key Concepts

- **Automatic Capture**: Trajectories are created when agent runs complete (opt-in)
- **Rich Metadata**: Captures timing, tokens, tools, models, and execution context
- **Turn Structure**: Groups messages by LLM inference step for clear decision boundaries
- **Outcome Scoring**: Heuristic-based initial scoring (success/partial/failure) with optional LLM refinement
- **Semantic Search**: Vector embeddings enable finding similar past experiences
- **Continual Learning**: Agents can search their own history to improve over time

## Architecture

```
┌─────────────────┐
│  Agent Run      │
│  (Run+Steps+    │
│   Messages)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Trajectory      │─────▶│ Trajectory       │
│ Converter       │      │ (trajectory_data)│
└────────┬────────┘      └────────┬─────────┘
         │                        │
         │                        ▼
         │               ┌──────────────────┐
         │               │ LLM Processing   │
         │               │ - Summary        │
         │               │ - Score          │
         │               │ - Embedding      │
         │               └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────┐
│         PostgreSQL + pgvector           │
│  (trajectories table with embeddings)   │
└─────────────────────────────────────────┘
```

## Trajectory Data Structure

A trajectory contains three main sections:

### 1. Metadata

High-level execution information:

```json
{
  "start_time": "2025-01-01T10:00:00+00:00",
  "end_time": "2025-01-01T10:05:30+00:00",
  "duration_ns": 330000000000,
  "status": "completed",
  "stop_reason": "end_turn",
  "step_count": 3,
  "message_count": 6,
  "tools_used": ["search_memory", "archival_memory_search"],
  "input_tokens": 1523,
  "output_tokens": 342,
  "total_tokens": 1865,
  "models": ["gpt-4"],
  "run_type": "send_message_streaming"
}
```

### 2. Turns

Chronological sequence of LLM inference steps:

```json
{
  "turns": [
    {
      "step_id": "step-abc123",
      "model": "gpt-4",
      "input_tokens": 512,
      "output_tokens": 128,
      "stop_reason": "tool_calls",
      "messages": [
        {
          "message_id": "msg-1",
          "role": "user",
          "timestamp": "2025-01-01T10:00:10+00:00",
          "content": [{"type": "text", "text": "Tell me about..."}]
        },
        {
          "message_id": "msg-2",
          "role": "assistant",
          "timestamp": "2025-01-01T10:00:15+00:00",
          "content": [{"type": "text", "text": "Let me search..."}],
          "tool_calls": [
            {
              "id": "call-1",
              "type": "function",
              "function": {
                "name": "search_memory",
                "arguments": "{\"query\": \"topic\"}"
              }
            }
          ]
        }
      ]
    }
  ]
}
```

### 3. Outcome

Automatic assessment of execution quality:

```json
{
  "outcome": {
    "type": "success",
    "confidence": 0.8,
    "reasoning": [
      "Run completed successfully",
      "Agent naturally ended turn",
      "High user engagement (4 user messages)"
    ]
  }
}
```

## Automatic Capture

### Enabling Capture

Set the environment variable to enable automatic trajectory creation:

```bash
export ENABLE_TRAJECTORY_CAPTURE=true
```

Or in Docker:

```yaml
environment:
  - ENABLE_TRAJECTORY_CAPTURE=true
```

### When Trajectories Are Created

Trajectories are automatically created when:
- An agent run completes (status: `completed`, `failed`, or `cancelled`)
- The `ENABLE_TRAJECTORY_CAPTURE` environment variable is `true`
- The run has at least one step and one message

The creation happens asynchronously and failures are logged but don't affect the run.

### What Gets Captured

**Included:**
- All messages (user, assistant, tool, system)
- Step metadata (model, tokens, stop reason)
- Tool calls and arguments
- Run timing and duration
- Outcome heuristics

**Not Included:**
- Raw LLM responses (only final messages)
- Internal state transitions
- Database queries
- Network requests

## LLM Processing

After creation, trajectories can be processed by an LLM to generate:

### 1. Searchable Summary

A natural language summary optimized for semantic search:

```python
# Automatically generated
summary = """
User requested a science fiction story about AI consciousness.
Agent approached by asking clarifying questions about themes,
then generated a structured narrative with world-building.
Execution was successful with high engagement and positive feedback.
"""
```

### 2. Outcome Score

Quality rating from 0-1 with reasoning:

```python
{
  "outcome_score": 0.85,
  "score_reasoning": "Task completed successfully with multiple"
                     "iterations. User expressed satisfaction and "
                     "requested follow-up content."
}
```

### 3. Vector Embedding

Embedding of the summary for similarity search (padded to 4096 dimensions).

### Processing API

```bash
# Trigger LLM processing
POST /v1/trajectories/{trajectory_id}/process

# Returns processed trajectory with summary, score, and embedding
{
  "id": "trajectory-123",
  "searchable_summary": "User requested...",
  "outcome_score": 0.85,
  "score_reasoning": "Task completed...",
  ...
}
```

## Searching Trajectories

### Semantic Search

Find trajectories similar to a query:

```bash
POST /v1/trajectories/search
{
  "query": "user wants story about AI and consciousness",
  "agent_id": "agent-123",  # optional
  "min_score": 0.7,          # filter by quality
  "limit": 5
}
```

Returns trajectories ordered by similarity with scores:

```json
{
  "results": [
    {
      "trajectory": {
        "id": "trajectory-456",
        "searchable_summary": "User requested sci-fi story...",
        "outcome_score": 0.85,
        "data": {...}
      },
      "similarity": 0.92
    }
  ]
}
```

### Filtering

List trajectories with filters:

```bash
GET /v1/trajectories/?agent_id=agent-123&min_score=0.7&limit=20
```

## REST API Reference

### Create Trajectory

```bash
POST /v1/trajectories/
{
  "agent_id": "agent-123",
  "data": {
    "run_id": "run-456",
    "metadata": {...},
    "turns": [...],
    "outcome": {...}
  }
}
```

### Get Trajectory

```bash
GET /v1/trajectories/{trajectory_id}
```

### Update Trajectory

```bash
PATCH /v1/trajectories/{trajectory_id}
{
  "data": {...}  # Update trajectory data
}
```

### Delete Trajectory

```bash
DELETE /v1/trajectories/{trajectory_id}
```

### Process Trajectory

```bash
POST /v1/trajectories/{trajectory_id}/process
```

Generates summary, score, and embedding using LLM.

### Search Trajectories

```bash
POST /v1/trajectories/search
{
  "query": "semantic search query",
  "agent_id": "agent-123",      # optional
  "min_score": 0.0,              # optional
  "max_score": 1.0,              # optional
  "limit": 10
}
```

## Database Schema

### Trajectories Table

```sql
CREATE TABLE trajectories (
  id VARCHAR PRIMARY KEY,
  agent_id VARCHAR NOT NULL,
  organization_id VARCHAR,

  -- Core data
  data JSON NOT NULL,

  -- LLM-processed fields
  searchable_summary TEXT,
  outcome_score FLOAT,
  score_reasoning TEXT,
  embedding VECTOR(4096),  -- pgvector for similarity search

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE,
  updated_at TIMESTAMP WITH TIME ZONE,

  -- Indexes
  FOREIGN KEY (agent_id) REFERENCES agents(id),
  INDEX idx_agent_score (agent_id, outcome_score),
  INDEX idx_embedding USING ivfflat (embedding vector_cosine_ops)
);
```

### Vector Search Index

pgvector provides efficient similarity search:

```sql
CREATE INDEX trajectories_embedding_idx
ON trajectories
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Implementation Details

### Converter Service

`TrajectoryConverter` transforms Run+Steps+Messages into trajectory format:

```python
from letta.services.trajectory_converter import TrajectoryConverter

converter = TrajectoryConverter()
trajectory = await converter.from_run(
    run=run,           # Run ORM object
    steps=steps,       # List[Step]
    messages=messages  # List[Message]
)
```

### Integration with RunManager

Trajectories are captured in `RunManager.update_run_by_id_async()`:

```python
# After run completes
if is_terminal_update:
    if os.getenv("ENABLE_TRAJECTORY_CAPTURE", "false").lower() == "true":
        await self._create_trajectory_from_run(run_id=run_id, actor=actor)
```

### Outcome Heuristics

Initial outcome determination uses heuristics:

- **Success** (0.7-0.8): Completed with natural end
- **Partial Success** (0.5): Hit token/time limits
- **Failure** (0.8-0.9): Error or cancellation
- **Unknown** (0.5): Ambiguous signals

Confidence adjusted by:
- User engagement (message count)
- Tool usage (active problem-solving)
- Stop reason analysis

### Embedding Padding

Embeddings are padded to `MAX_EMBEDDING_DIM` (4096) to match database schema:

```python
import numpy as np
from letta.constants import MAX_EMBEDDING_DIM

# Pad 1536-dim embedding to 4096
embedding_array = np.array(embedding)
padded = np.pad(
    embedding_array,
    (0, MAX_EMBEDDING_DIM - len(embedding)),
    mode="constant"
)
```

## Configuration

### Environment Variables

```bash
# Enable automatic capture (default: false)
ENABLE_TRAJECTORY_CAPTURE=true

# OpenAI API key for LLM processing
OPENAI_API_KEY=sk-...

# Database connection (for pgvector)
LETTA_PG_URI=postgresql://user:pass@localhost:5432/letta
```

### Docker Compose

```yaml
services:
  letta_server:
    environment:
      - ENABLE_TRAJECTORY_CAPTURE=true
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LETTA_PG_URI=postgresql://letta:letta@pgvector_db:5432/letta
```

## Testing

### Running Tests

```bash
# Run trajectory converter tests
uv run pytest tests/test_trajectory_converter.py -v

# Run with coverage
uv run pytest tests/test_trajectory_converter.py --cov=letta.services.trajectory_converter
```

### Manual Testing

1. Enable trajectory capture:
```bash
export ENABLE_TRAJECTORY_CAPTURE=true
```

2. Run an agent conversation:
```bash
uv run letta run --agent your-agent-id
```

3. Check trajectory was created:
```bash
curl http://localhost:8283/v1/trajectories/?agent_id=your-agent-id
```

4. Process the trajectory:
```bash
curl -X POST http://localhost:8283/v1/trajectories/{id}/process
```

5. Search for similar trajectories:
```bash
curl -X POST http://localhost:8283/v1/trajectories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "limit": 5}'
```

## Best Practices

### When to Enable Capture

**Enable for:**
- Production agents where learning from history is valuable
- Evaluation and analysis of agent behavior
- Building training datasets from real interactions
- Debugging complex multi-turn conversations

**Disable for:**
- High-throughput systems where storage is a concern
- Development/testing with throwaway conversations
- Privacy-sensitive applications (trajectories contain full messages)

### Storage Considerations

Each trajectory averages:
- **Data field**: 5-50 KB (depends on conversation length)
- **Summary**: 1-2 KB
- **Embedding**: 16 KB (4096 floats × 4 bytes)
- **Total**: ~20-70 KB per trajectory

For 10,000 trajectories: ~200-700 MB

Consider implementing retention policies for large-scale deployments.

### Performance Impact

Trajectory creation is async and non-blocking:
- **Capture**: <10ms (conversion + DB insert)
- **LLM Processing**: 2-5 seconds (can be done in background)
- **Search**: 50-200ms (pgvector index lookup)

Capture failures don't affect agent execution.

## Troubleshooting

### Trajectories Not Being Created

1. Check environment variable:
```bash
echo $ENABLE_TRAJECTORY_CAPTURE
```

2. Check logs for errors:
```bash
docker logs letta-server | grep trajectory
```

3. Verify database migration ran:
```bash
uv run alembic current
# Should show migration: 56e2a174be96
```

### Search Returns No Results

1. Check if trajectories have embeddings:
```sql
SELECT COUNT(*) FROM trajectories WHERE embedding IS NOT NULL;
```

2. Process unprocessed trajectories:
```bash
curl -X POST http://localhost:8283/v1/trajectories/{id}/process
```

3. Verify pgvector extension:
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Embedding Dimension Mismatch

Error: `expected 4096 dimensions, not 1536`

**Cause**: Embedding model changed or padding not applied

**Fix**: Ensure `TrajectoryProcessor.generate_embedding()` pads to `MAX_EMBEDDING_DIM`

## Future Enhancements

Potential improvements being considered:

- **Agent Tools**: Function set for agents to search their own trajectories
- **Automatic Tagging**: LLM-based categorization and labeling
- **Pattern Analysis**: Aggregate statistics on success/failure patterns
- **Retention Policies**: Automatic cleanup of old/low-value trajectories
- **Cost Optimization**: Batching LLM processing, smaller embeddings
- **Privacy Controls**: PII redaction, user consent tracking

## Contributing

When contributing trajectory-related features:

1. Follow the existing patterns in `trajectory_converter.py`
2. Add comprehensive tests (see `test_trajectory_converter.py`)
3. Update this documentation
4. Ensure backward compatibility (new fields optional)
5. Run the full test suite before submitting

## Additional Resources

- **Source Code**: `letta/services/trajectory_*.py`
- **Tests**: `tests/test_trajectory_converter.py`
- **Migration**: `alembic/versions/56e2a174be96_add_trajectories_table.py`
- **API Router**: `letta/server/rest_api/routers/v1/trajectories.py`
