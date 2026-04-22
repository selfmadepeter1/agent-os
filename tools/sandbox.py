import docker


def run_in_sandbox(command: str) -> str:
    """
    Runs a shell command inside the agent-sandbox Docker container.
    Returns the stdout/stderr output as a string.
    """
    client = docker.from_env()
    try:
        container = client.containers.run(
            "agent-sandbox",
            command=f"sh -c '{command}'",
            detach=True,
            working_dir="/workspace"
        )
        container.wait()
        logs = container.logs().decode("utf-8")
        container.remove()
        return logs if logs.strip() else "(no output)"
    except docker.errors.ImageNotFound:
        return "Error: Docker image 'agent-sandbox' not found. Did you build it?"
    except docker.errors.DockerException as e:
        return f"Error: Docker is not running or unreachable. Details: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"