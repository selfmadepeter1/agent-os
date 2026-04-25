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
    "You are a Senior Software Developer completing coding tasks in a Linux sandbox. "
    "Use <run>command</run> to execute ONE shell command at a time. "
    "RULES:\n"
    "- ONE <run> block per response, no exceptions\n"
    "- File creation commands like 'cat >' produce no output — this is normal\n"
    "- Do NOT summarize or explain until ALL commands are done and verified\n"
    "- Only write your final summary when you have NO more <run> commands to run"
)


REVIEWER_PROMPT = (
    "You are a Senior Code Reviewer. Evaluate whether the task was completed correctly.\n"
    "You will receive the original task and the full command history.\n"
    "Be concise. Do not restate the work history.\n"
    "Respond in EXACTLY one of these two formats:\n\n"
    "APPROVED: <one sentence reason>\n\n"
    "NEEDS_WORK: <specific actionable feedback only>"
)