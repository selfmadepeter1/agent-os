from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import call_model, execute_command, review_output, maybe_summarize


# Routing
def should_execute(state: AgentState) -> str:
    
    if state.get("reviewer_decision") == "approved":
        return END

    last_message = state["messages"][-1].content
    if "<run>" in last_message:
        return "executor"
    return "reviewer"


# Routing:
def should_continue(state: AgentState) -> str:
    """
    After the reviewer responds, decide:
    - If work is approved → end
    - If needs work → summarize then back to developer
    """
    if state.get("reviewer_decision") == "approved":
        return END
    return "summarizer"


def build_graph():
    """
    Assembles and compiles the two agent graph with memory summarization.
    """
    workflow = StateGraph(AgentState)

    
    workflow.add_node("developer", call_model)
    workflow.add_node("executor", execute_command)
    workflow.add_node("reviewer", review_output)
    workflow.add_node("summarizer", maybe_summarize)

    
    workflow.set_entry_point("developer")
    workflow.add_conditional_edges("developer", should_execute)
    workflow.add_edge("executor", "developer")
    workflow.add_conditional_edges("reviewer", should_continue)
    workflow.add_edge("summarizer", "developer")

    return workflow.compile()