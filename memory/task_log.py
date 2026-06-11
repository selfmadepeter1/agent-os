import json
import os
import time
from datetime import datetime
from typing import Optional, List

from memory.vector_store import VectorStore

TASK_LOG_PATH = "task_log.json"

# Singleton vector store — created once, reused across calls
_vector_store: Optional[VectorStore] = None


def _get_vector_store() -> VectorStore:
    """
    Lazy-initialises the vector store on first use.
    Avoids loading ChromaDB at import time if memory isn't used.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def load_task_log() -> list:
    """
    Loads the full task log from disk.
    Returns an empty list if no log exists yet.
    """
    if not os.path.exists(TASK_LOG_PATH):
        return []

    with open(TASK_LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_task(task: str, terminal_history: list, outcome: str):
    """
    Appends a completed task record to the JSON log
    and stores its embedding in ChromaDB.
    """
    log = load_task_log()

    # Unique ID linking JSON record and ChromaDB embedding
    task_id = f"task_{int(time.time() * 1000)}"

    record = {
        "id": task_id,
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "outcome": outcome,
        "commands_run": len(terminal_history),
        "terminal_history": terminal_history,
    }

    log.append(record)

    with open(TASK_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    # Store embedding in ChromaDB
    vector_store = _get_vector_store()
    vector_store.store_task_embedding(task_id, task)

    print(
        f"--- Task saved to memory "
        f"({len(log)} total tasks, "
        f"{vector_store.count()} embeddings) ---"
    )


def _lookup_tasks_by_ids(task_ids: List[str]) -> List[dict]:
    """
    Given a list of task IDs, returns full records from the JSON log
    in the same order as the IDs.
    """
    log = load_task_log()
    log_by_id = {record["id"]: record for record in log if "id" in record}

    return [
        log_by_id[task_id]
        for task_id in task_ids
        if task_id in log_by_id
    ]


def get_relevant_context(task: str, max_results: int = 3) -> Optional[str]:
    """
    Returns context from semantically similar past tasks.
    Falls back to recency if no embeddings exist yet
    (e.g. for tasks logged before Phase 6).
    """
    log = load_task_log()
    if not log:
        return None

    vector_store = _get_vector_store()

    # ── Try semantic search first ──────────────────────────────────────
    similar_ids = vector_store.find_similar_tasks(task, max_results)

    if similar_ids:
        relevant_tasks = _lookup_tasks_by_ids(similar_ids)
        retrieval_method = "semantic similarity"
    else:
        # Fallback for backwards compatibility with pre-Phase 6 logs
        relevant_tasks = log[-max_results:]
        retrieval_method = "recency (no embeddings yet)"

    if not relevant_tasks:
        return None

    context_lines = [
        f"Here are past tasks relevant to your current goal "
        f"(retrieved by {retrieval_method}):\n"
    ]

    for record in relevant_tasks:
        outcome = record.get("outcome", "unknown")
        commands = record.get("commands_run", 0)
        timestamp = record.get("timestamp", "")[:10]
        context_lines.append(
            f"- [{timestamp}] Task: \"{record['task']}\" "
            f"→ Outcome: {outcome} ({commands} commands run)"
        )

    return "\n".join(context_lines)