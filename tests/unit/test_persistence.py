from pathlib import Path

from qwen_audio_agent.runtime.persistence import open_runtime_persistence


async def test_runtime_store_survives_reopen(tmp_path: Path) -> None:
    namespace = ("authorization-index",)

    async with open_runtime_persistence(tmp_path) as persistence:
        await persistence.store.aput(namespace, "auth-1", {"thread_id": "work:task_1"})

    async with open_runtime_persistence(tmp_path) as persistence:
        item = await persistence.store.aget(namespace, "auth-1")

    assert item is not None
    assert item.value == {"thread_id": "work:task_1"}
