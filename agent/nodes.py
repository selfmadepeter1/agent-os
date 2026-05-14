import re
import shlex
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from tools import Sandbox

from config import (
    MODEL_NAME, TEMPERATURE,
    DEVELOPER_PROMPT, REVIEWER_PROMPT,
    PLANNER_PROMPT, TASK_MANAGER_PROMPT,
    BLOCKED_COMMANDS, MAX_RETRIES
)
from agent.state import AgentState
from memory import should_summarize, summarize_messages



model = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)


# ── Node 1: Planner ────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    """
    Reads the high-level goal and decomposes it into an ordered
    list of subtasks. Only runs once at the start of a task.
    """
    
    if state.get("task_plan"):
        return {}

    response = model.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Break down this goal into subtasks:\n{state['original_task']}")
    ])

    # Parse the numbered list into SubTask dicts
    lines = response.content.strip().split("\n")
    task_plan = []
    for i, line in enumerate(lines):
        
        description = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if description:
            task_plan.append({
                "id": i + 1,
                "description": description,
                "status": "pending"
            })

    print(f"\n--- [planner] Breaking into {len(task_plan)} subtasks ---")
    for subtask in task_plan:
        print(f"  [{subtask['id']}] {subtask['description']}")

    
    if task_plan:
        task_plan[0]["status"] = "in_progress"

    return {
        "task_plan": task_plan,
        "current_task_index": 0,
        "retry_count": 0,
        "messages": [HumanMessage(
            content=f"Your first subtask is:\n{task_plan[0]['description']}"
        )]
    }



def call_model(state: AgentState) -> dict:
    """
    The Developer agent. Thinks about the current subtask and decides
    what command to run next, or summarizes when the subtask is complete.
    """
    response = model.invoke(
        [SystemMessage(content=DEVELOPER_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}



def execute_command(state: AgentState) -> dict:
    """
    Reads the last message, extracts any <run> command, and executes
    it in the persistent sandbox container.
    """
    last_message = state["messages"][-1].content
    match = re.search(r"<run>(.*?)</run>", last_message, re.DOTALL)

    if not match:
        return {}

    command = match.group(1).strip()

    try:
        tokens = shlex.split(command.lower())
    except ValueError:
        tokens = command.lower().split()

    if BLOCKED_COMMANDS.intersection(tokens):
        return {
            "messages": [HumanMessage(content="Error: Command blocked for safety.")]
        }

    print(f"\n--- Running inside Docker: {command} ---")
    sandbox = state["sandbox"]
    result = sandbox.run(command)

    return {
        "messages": [HumanMessage(content=f"Terminal Output:\n{result}")],
        "terminal_history": [f"$ {command}\n{result}"],
    }



def review_output(state: AgentState) -> dict:
    """
    The Reviewer agent. Evaluates whether the current subtask
    was completed correctly.
    """
    task_plan = state.get("task_plan", [])
    current_index = state.get("current_task_index", 0)
    current_subtask = (
        task_plan[current_index]["description"]
        if task_plan else state["original_task"]
    )

    terminal_summary = "\n".join(state["terminal_history"])

    review_request = (
        f"=== Overall Goal ===\n{state['original_task']}\n\n"
        f"=== Current Subtask ===\n{current_subtask}\n\n"
        f"=== Commands Run ===\n{terminal_summary}\n\n"
        f"Did the developer complete the current subtask correctly?"
    )

    response = model.invoke([
        SystemMessage(content=REVIEWER_PROMPT),
        HumanMessage(content=review_request)
    ])

    verdict = response.content.strip()
    print(f"\n--- [reviewer] ---\n{verdict}")

    if verdict.upper().startswith("APPROVED"):
        return {"reviewer_decision": "approved"}
    else:
        return {
            "messages": [HumanMessage(content=f"Reviewer feedback:\n{verdict}")],
            "reviewer_decision": "needs_work",
            "retry_count": state.get("retry_count", 0) + 1
        }



def task_manager_node(state: AgentState) -> dict:
    """
    Runs after reviewer approves. Marks current subtask as done,
    loads the next one, or signals completion if all done.
    """
    task_plan = state.get("task_plan", [])
    current_index = state.get("current_task_index", 0)

    
    task_plan[current_index]["status"] = "done"

    next_index = current_index + 1

    
    if next_index >= len(task_plan):
        print("\n--- [task_manager] All subtasks complete ---")
        return {
            "task_plan": task_plan,
            "current_task_index": next_index,
            "reviewer_decision": "",
        }

    
    task_plan[next_index]["status"] = "in_progress"
    next_subtask = task_plan[next_index]["description"]

    print(f"\n--- [task_manager] Moving to subtask {next_index + 1}: {next_subtask} ---")

    return {
        "task_plan": task_plan,
        "current_task_index": next_index,
        "retry_count": 0,
        "reviewer_decision": "",
        "messages": [HumanMessage(
            content=f"Subtask {next_index + 1} of {len(task_plan)}:\n{next_subtask}"
        )]
    }



def maybe_summarize(state: AgentState) -> dict:
    """
    Checks if the conversation is getting long and compresses
    it if so. Runs silently — returns nothing if not needed.
    """
    messages = state["messages"]

    if should_summarize(messages):
        print("\n--- Summarizing conversation history ---")
        compressed = summarize_messages(messages)
        return {
            "messages": compressed,
            "conversation_summary": "summarized"
        }

    return {}