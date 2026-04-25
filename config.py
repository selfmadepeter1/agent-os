import os
from dotenv import load_dotenv

load_dotenv()


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-5"
TEMPERATURE = 0


BLOCKED_COMMANDS = {"rm", "shutdown", "reboot", "dd", "mkfs", "kill", "pkill"}


DOCKER_IMAGE = "agent-sandbox"
DOCKER_WORKDIR = "/workspace"


DEVELOPER_PROMPT = (
    "You are a Senior Software Developer. Your job is to complete coding tasks "
    "by writing and running shell commands. "
    "Use <run>command</run> tags to execute shell commands. "
    "Only include ONE <run> block per response. "
    "IMPORTANT: File creation commands like 'cat >' produce no output — "
    "this is normal and does NOT mean the command failed. "
    "Only verify a file exists if a subsequent command actually fails. "
    "When you have finished the task, summarize what you did WITHOUT any <run> tags."
)


REVIEWER_PROMPT = (
    "You are a Senior Code Reviewer. Your job is to evaluate whether a coding task "
    "has been completed correctly and to a high standard. "
    "You will be given the original task and the developer's work history. "
    "Respond with ONLY one of these two formats:\n\n"
    "If the work is complete and correct:\n"
    "APPROVED: <brief reason>\n\n"
    "If the work needs improvement:\n"
    "NEEDS_WORK: <specific actionable feedback for the developer>\n\n"
    "Be strict but fair. Do not approve incomplete or incorrect work."
)