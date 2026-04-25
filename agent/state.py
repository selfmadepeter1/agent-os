from typing import Annotated, TypedDict, List, Any
from operator import add
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    
    messages: Annotated[List[BaseMessage], add]

    
    terminal_history: Annotated[List[str], add]

    
    reviewer_decision: str

    
    sandbox: Any