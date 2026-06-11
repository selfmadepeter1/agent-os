from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from tools import Sandbox
from memory import save_task, get_relevant_context
from observability import RunTracker

load_dotenv()


def run(task: str):
    app = build_graph()
    sandbox = Sandbox()
    tracker = RunTracker()

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

    terminal_history = []
    final_decision = "unknown"
    final_task_plan = []

    try:
        sandbox.start()

        inputs = {
            "messages": initial_messages,
            "terminal_history": [],
            "reviewer_decision": "",
            "sandbox": sandbox,
            "original_task": task,
            "conversation_summary": "",
            "task_plan": [],
            "current_task_index": 0,
            "retry_count": 0,
            "tracker": tracker,
        }

        try:
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

                if state.get("task_plan"):
                    final_task_plan = state["task_plan"]

        except Exception as stream_error:
            print(f"\n--- Stream ended early: {stream_error} ---")
            print("--- Continuing to post-run summary ---")

    finally:
        sandbox.stop()

    try:
        

        if final_task_plan:
            print("\n=== Task Plan Summary ===")
            for subtask in final_task_plan:
                icon = "✓" if subtask["status"] == "done" else "✗"
                print(f"  [{icon}] {subtask['description']}")

        
        tracker.print_summary()
        

        
        save_task(task, terminal_history, final_decision)
        

        print("\n=== Terminal History ===")
        for entry in terminal_history:
            print(entry)

        

    except Exception as e:
        import traceback
        print(f"\n--- POST-RUN ERROR: {e} ---")
        traceback.print_exc()


if __name__ == "__main__":
    task = input("What task should the agent perform? ")
    run(task)