from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from tools import Sandbox

load_dotenv()


def run(task: str):
    """
    Runs the agent on a given task string.
    Creates a persistent sandbox for the task lifetime.
    """
    app = build_graph()
    sandbox = Sandbox()

    try:
        sandbox.start()

        inputs = {
            "messages": [HumanMessage(content=task)],
            "terminal_history": [],
            "reviewer_decision": "",
            "sandbox": sandbox,
        }

        terminal_history = []

        for event in app.stream(inputs):
            node_name = list(event.keys())[0]
            state = event[node_name]

            print(f"\n--- [{node_name}] ---")

            if state.get("messages"):
                print(state["messages"][-1].content)

            if state.get("terminal_history"):
                terminal_history.extend(state["terminal_history"])

    finally:
        
        sandbox.stop()

    print("\n=== Terminal History ===")
    for entry in terminal_history:
        print(entry)


if __name__ == "__main__":
    task = input("What task should the agent perform? ")
    run(task)