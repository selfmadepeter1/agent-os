import re
import shlex
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from tools import Sandbox
from memory import should_summarize, summarize_messages

from config import (
    MODEL_NAME, TEMPERATURE,
    DEVELOPER_PROMPT, REVIEWER_PROMPT,
    BLOCKED_COMMANDS
)
from agent.state import AgentState


#  Model instance 
model = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)


#  Node 1: Developer 
def call_model(state: AgentState) -> dict:
    """
    The Developer agent. Thinks about the task and decides what
    command to run next, or summarizes when the task is complete.
    """
    response = model.invoke(
        [SystemMessage(content=DEVELOPER_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


#  Node 2: Executor 
def execute_command(state: AgentState) -> dict:
    """
    Reads the last message, extracts any <run> command, and executes it
    in the persistent sandbox container.
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


# Node 3
def review_output(state: AgentState) -> dict:
    """
    The Reviewer agent. Reads the original task, full conversation and
    terminal history, then decides if the work is complete and correct.
    """
    
    original_task = state["messages"][0].content

    # Build the full terminal log
    terminal_summary = "\n".join(state["terminal_history"])

    review_request = (
        f"=== Original Task ===\n{original_task}\n\n"
        f"=== Commands Run & Output ===\n{terminal_summary}\n\n"
        f"Review whether the original task has been completed correctly "
        f"and give your verdict."
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
            "reviewer_decision": "needs_work"
        }

def maybe_summarize(state: AgentState) -> dict:
    """
    Checks if the conversation is getting long and compresses it if so.
    Runs silently — if no summarization needed, returns nothing.
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