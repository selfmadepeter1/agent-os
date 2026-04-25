import json
import os
from datetime import datetime
from typing import Optional

TASK_LOG_PATH = "task_log.json"


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
    Appends a completed task record to the log on disk.

    outcome should be "approved" or "needs_work"
    """
    log = load_task_log()

    record = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "outcome": outcome,
        "commands_run": len(terminal_history),
        "terminal_history": terminal_history,
    }

    log.append(record)

    with open(TASK_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"--- Task saved to memory ({len(log)} total tasks logged) ---")


def get_relevant_context(task: str, max_results: int = 3) -> Optional[str]:
    
    log = load_task_log()

    if not log:
        return None

    
    recent = log[-max_results:]

    context_lines = ["Here are recent tasks you have completed:\n"]

    for record in recent:
        context_lines.append(
            f"- [{record['timestamp'][:10]}] Task: \"{record['task']}\" "
            f"→ Outcome: {record['outcome']} "
            f"({record['commands_run']} commands run)"
        )

    return "\n".join(context_lines)