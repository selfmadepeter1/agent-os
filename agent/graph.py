from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    planner_node,
    call_model,
    execute_command,
    review_output,
    task_manager_node,
    maybe_summarize
)



def after_planner(state: AgentState) -> str:
    """
    After planner runs, always go to developer.
    If plan was already set, planner returned {} so we still proceed.
    """
    return "developer"


#
def should_execute(state: AgentState) -> str:
    """
    After developer responds:
    - If response contains a command → executor
    - If no command → reviewer
    """
    last_message = state["messages"][-1].content
    if "<run>" in last_message:
        return "executor"
    return "reviewer"



def after_review(state: AgentState) -> str:
    """
    After reviewer responds:
    - If approved → task_manager to advance the plan
    - If needs_work and retries exceeded → task_manager to skip
    - If needs_work → summarizer then back to developer
    """
    from config import MAX_RETRIES

    if state.get("reviewer_decision") == "approved":
        return "task_manager"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"\n--- Max retries reached, skipping subtask ---")
        # Force approve so task manager advances
        state["reviewer_decision"] = "approved"
        return "task_manager"

    return "summarizer"



def after_task_manager(state: AgentState) -> str:
    """
    After task manager runs:
    - If more subtasks remain → back to developer
    - If all done → END
    """
    task_plan = state.get("task_plan", [])
    current_index = state.get("current_task_index", 0)

    if current_index >= len(task_plan):
        return END
    return "developer"


def build_graph():
    """
    Assembles and compiles the full Phase 4 agent graph.
    """
    workflow = StateGraph(AgentState)

    
    workflow.add_node("planner",      planner_node)
    workflow.add_node("developer",    call_model)
    workflow.add_node("executor",     execute_command)
    workflow.add_node("reviewer",     review_output)
    workflow.add_node("task_manager", task_manager_node)
    workflow.add_node("summarizer",   maybe_summarize)

    
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner",      after_planner)
    workflow.add_conditional_edges("developer",    should_execute)
    workflow.add_edge("executor",                  "developer")
    workflow.add_conditional_edges("reviewer",     after_review)
    workflow.add_edge("summarizer",                "developer")
    workflow.add_conditional_edges("task_manager", after_task_manager)

    return workflow.compile()