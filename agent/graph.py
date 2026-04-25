from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import call_model, execute_command, review_output


# graph routes: Developer to Executor or Reviewer 
def should_execute(state: AgentState) -> str:
    """
    After the developer responds, decide:
    - If already approved → end immediately
    - If it wants to run a command → executor
    - If it's done thinking → reviewer
    """
    # If reviewer already approved, don't loop back
    if state.get("reviewer_decision") == "approved":
        return END

    last_message = state["messages"][-1].content
    if "<run>" in last_message:
        return "executor"
    return "reviewer"



def should_continue(state: AgentState) -> str:
    """
    After the reviewer responds, decide:
    - If work is approved → end
    - If needs work → back to developer
    """
    if state.get("reviewer_decision") == "approved":
        return END
    return "developer"


def build_graph():
    """
    Assembles and compiles the two-agent graph.
    """
    workflow = StateGraph(AgentState)

    
    workflow.add_node("developer", call_model)
    workflow.add_node("executor", execute_command)
    workflow.add_node("reviewer", review_output)

    
    workflow.set_entry_point("developer")
    workflow.add_conditional_edges("developer", should_execute)
    workflow.add_edge("executor", "developer")
    workflow.add_conditional_edges("reviewer", should_continue)

    return workflow.compile()