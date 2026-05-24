import re
import shlex
import time
from anthropic import RateLimitError
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


def _invoke_and_track(state: AgentState, node_name: str, messages: list):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = model.invoke(messages)
            duration = time.time() - start
            break
        except RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"\n--- Rate limit hit. Waiting {wait}s before retry ---")
            time.sleep(wait)
            if attempt == max_retries - 1:
                raise

    prompt_tokens = 0
    completion_tokens = 0
    cache_read_tokens = 0

    usage = getattr(response, "usage_metadata", None)
    if usage:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)

    if prompt_tokens == 0:
        response_meta = getattr(response, "response_metadata", {})
        usage_meta = response_meta.get("usage", {})
        prompt_tokens = usage_meta.get("input_tokens", 0)
        completion_tokens = usage_meta.get("output_tokens", 0)
        cache_read_tokens = usage_meta.get("cache_read_input_tokens", 0)

    tracker = state.get("tracker")
    if tracker:
        tracker.record(
            node_name=node_name,
            duration=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens
        )

    return response


def planner_node(state: AgentState) -> dict:
    if state.get("task_plan"):
        return {}

    response = _invoke_and_track(state, "planner", [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(
            content=f"Break down this goal into subtasks:\n{state['original_task']}"
        )
    ])

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
    response = _invoke_and_track(
        state, "developer",
        [SystemMessage(content=DEVELOPER_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def execute_command(state: AgentState) -> dict:
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

    response = _invoke_and_track(state, "reviewer", [
        SystemMessage(content=REVIEWER_PROMPT),
        HumanMessage(content=review_request)
    ])

    verdict = response.content.strip()
    print(f"\n--- [reviewer] ---\n{verdict}")

    if verdict.upper().startswith("APPROVED"):
        tracker = state.get("tracker")
        if tracker:
            tracker.mark_subtask_complete()
        return {"reviewer_decision": "approved"}
    else:
        tracker = state.get("tracker")
        if tracker:
            tracker.mark_subtask_failed()
        return {
            "messages": [HumanMessage(content=f"Reviewer feedback:\n{verdict}")],
            "reviewer_decision": "needs_work",
            "retry_count": state.get("retry_count", 0) + 1
        }


def task_manager_node(state: AgentState) -> dict:
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

    print(
        f"\n--- [task_manager] Moving to subtask "
        f"{next_index + 1}: {next_subtask} ---"
    )

    return {
        "task_plan": task_plan,
        "current_task_index": next_index,
        "retry_count": 0,
        "reviewer_decision": "",
        "messages": [HumanMessage(
            content=(
                f"Subtask {next_index + 1} of {len(task_plan)}:\n{next_subtask}"
            )
        )]
    }


def maybe_summarize(state: AgentState) -> dict:
    messages = state["messages"]

    if should_summarize(messages):
        print("\n--- Summarizing conversation history ---")
        compressed = summarize_messages(messages)
        return {
            "messages": compressed,
            "conversation_summary": "summarized"
        }

    return {}