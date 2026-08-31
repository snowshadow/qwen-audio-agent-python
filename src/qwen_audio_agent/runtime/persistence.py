"""Durable LangGraph persistence owned by the Python Gateway runtime.

Graph checkpoints are execution snapshots. The Store is limited to small,
cross-thread runtime indexes. Neither replaces the task repository, delivery
lease, conversation transcript, or backend adapter's private session store.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from anyio import Path as AsyncPath
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore


@dataclass(frozen=True, slots=True)
class RuntimePersistence:
    """Persistence handles injected into compiled graphs."""

    checkpointer: AsyncSqliteSaver
    store: AsyncSqliteStore


@asynccontextmanager
async def open_runtime_persistence(root: Path) -> AsyncIterator[RuntimePersistence]:
    """Open and initialize the local durable checkpoint and runtime stores."""

    await AsyncPath(root).mkdir(parents=True, exist_ok=True, mode=0o700)
    checkpoint_path = root / "langgraph-checkpoints.sqlite3"
    store_path = root / "langgraph-store.sqlite3"
    async with (
        AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(store_path)) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        yield RuntimePersistence(checkpointer=checkpointer, store=store)
