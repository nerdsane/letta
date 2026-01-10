"""
Storage backends for OTS trajectories.

Provides:
- TrajectoryStore: Main facade for storing and retrieving trajectories
- LanceDBBackend: Default for context learning (vector search)
- SQLiteBackend: Simple file-based storage (no vector search)
- FileBackend: JSON file storage
- MemoryBackend: In-memory storage for testing

Recommended:
- Use LanceDBBackend for context learning (pip install ots[lancedb])
- Use SQLiteBackend for simple storage without semantic search
- Use PostgreSQLBackend for production scale (pip install ots[postgres])
"""

from ots.store.base import TrajectoryStore
from ots.store.sqlite import SQLiteBackend
from ots.store.file import FileBackend
from ots.store.memory import MemoryBackend

# LanceDB is optional - import lazily
try:
    from ots.store.lancedb import LanceDBBackend
except ImportError:
    LanceDBBackend = None  # type: ignore

# PostgreSQL is optional - import lazily
try:
    from ots.store.postgres import PostgresBackend
except ImportError:
    PostgresBackend = None  # type: ignore

__all__ = [
    "TrajectoryStore",
    "LanceDBBackend",
    "PostgresBackend",
    "SQLiteBackend",
    "FileBackend",
    "MemoryBackend",
]
