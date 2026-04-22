import os
from dotenv import load_dotenv

load_dotenv()


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-5"
TEMPERATURE = 0


BLOCKED_COMMANDS = {"rm", "shutdown", "reboot", "dd", "mkfs", "kill", "pkill"}


DOCKER_IMAGE = "agent-sandbox"
DOCKER_WORKDIR = "/workspace"

# ── Agent Behaviour ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Senior SRE. Use <run>command</run> tags to execute shell commands. "
    "Only include ONE <run> block per response. "
    "When you have enough information and no more commands are needed, "
    "summarize your findings WITHOUT any <run> tags."
)