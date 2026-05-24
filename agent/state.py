from typing import Annotated, TypedDict, List, Any
from operator import add
from langchain_core.messages import BaseMessage


class SubTask(TypedDict):
    id: int
    description: str
    status: str


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    terminal_history: Annotated[List[str], add]
    reviewer_decision: str
    sandbox: Any
    original_task: str
    conversation_summary: str
    task_plan: List[SubTask]
    current_task_index: int
    retry_count: int
    tracker: Any