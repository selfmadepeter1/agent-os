from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from typing import List
from config import MODEL_NAME, TEMPERATURE

# Can be changed on individual basis
RECENT_MESSAGES_TO_KEEP = 6

summarizer_model = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)

SUMMARIZER_PROMPT = (
    "You are a conversation summarizer. "
    "Given a list of messages from an AI coding session, "
    "produce a concise summary that captures: "
    "1) What the original task was, "
    "2) What steps were taken, "
    "3) What the current state is (what files exist, what works). "
    "Be factual and brief. This summary will replace the earlier "
    "messages to save context space."
)


def should_summarize(messages: List[BaseMessage], threshold: int = 20) -> bool:
    
    return len(messages) > threshold


def summarize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    
    if len(messages) <= RECENT_MESSAGES_TO_KEEP:
        return messages

    
    messages_to_summarize = messages[:-RECENT_MESSAGES_TO_KEEP]
    recent_messages = messages[-RECENT_MESSAGES_TO_KEEP:]

    # Build a readable transcript for the summarizer
    transcript = []
    for msg in messages_to_summarize:
        if isinstance(msg, HumanMessage):
            transcript.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            transcript.append(f"Assistant: {msg.content}")

    transcript_text = "\n\n".join(transcript)

    
    summary_response = summarizer_model.invoke([
        SystemMessage(content=SUMMARIZER_PROMPT),
        HumanMessage(content=f"Summarize this conversation:\n\n{transcript_text}")
    ])

    summary_message = HumanMessage(
        content=f"[Conversation Summary]\n{summary_response.content}"
    )

    return [summary_message] + recent_messages