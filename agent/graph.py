from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import call_model, execute_command


def should_continue(state: AgentState) -> str:
    """
    Router function — reads the brain's last message and decides
    whether to execute a command or end the run.
    """
    last_message = state["messages"][-1].content
    if "<run>" in last_message:
        return "executor"
    return END


def build_graph():
    """
    Assembles and compiles the agent graph.
    Returns a runnable app.
    """
    workflow = StateGraph(AgentState)

    
    workflow.add_node("brain", call_model)
    workflow.add_node("executor", execute_command)

    
    workflow.set_entry_point("brain")
    workflow.add_conditional_edges("brain", should_continue)
    workflow.add_edge("executor", "brain")

    return workflow.compile()