import re
import shlex
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from config import MODEL_NAME, TEMPERATURE, SYSTEM_PROMPT, BLOCKED_COMMANDS
from agent.state import AgentState
from tools import run_in_sandbox


#  Model instance
model = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)



def call_model(state: AgentState) -> dict:
    """
    Sends the full conversation history to the model and returns its response.
    This is the 'thinking' step — the agent decides what to do next.
    """
    response = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}



def execute_command(state: AgentState) -> dict:
    """
    Reads the last message, extracts any <run> command, and executes it
    in the Docker sandbox. Returns the terminal output as a new message.
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
    result = run_in_sandbox(command)

    return {
        "messages": [HumanMessage(content=f"Terminal Output:\n{result}")],
        "terminal_history": [f"$ {command}\n{result}"],
    }