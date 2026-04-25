from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from tools import Sandbox
from memory import save_task, get_relevant_context

load_dotenv()


def run(task: str):
    """
    Runs the agent on a given task string.
    Loads relevant past context before starting.
    Saves the completed task to memory after finishing.
    """
    app = build_graph()
    sandbox = Sandbox()

    
    past_context = get_relevant_context(task)

    
    if past_context:
        print("\n--- Loading memory context ---")
        print(past_context)
        initial_messages = [
            HumanMessage(content=past_context),
            HumanMessage(content=task),
        ]
    else:
        initial_messages = [HumanMessage(content=task)]

    try:
        sandbox.start()

        inputs = {
            "messages": initial_messages,
            "terminal_history": [],
            "reviewer_decision": "",
            "sandbox": sandbox,
            "original_task": task,
            "conversation_summary": "",
        }

        terminal_history = []
        final_decision = "unknown"

        for event in app.stream(inputs):
            node_name = list(event.keys())[0]
            state = event[node_name]

            print(f"\n--- [{node_name}] ---")

            if state.get("messages"):
                print(state["messages"][-1].content)

            if state.get("terminal_history"):
                terminal_history.extend(state["terminal_history"])

            
            if state.get("reviewer_decision"):
                final_decision = state["reviewer_decision"]

    finally:
        sandbox.stop()

    # 
    final_decision = inputs.get("reviewer_decision", "unknown")
    save_task(task, terminal_history, final_decision)

    print("\n=== Terminal History ===")
    for entry in terminal_history:
        print(entry)


if __name__ == "__main__":
    task = input("What task should the agent perform? ")
    run(task)