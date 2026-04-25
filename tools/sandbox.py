import docker


class Sandbox:
    """
    Manages a single Docker container for the lifetime of a task.
    Commands run sequentially against the same container,
    so files persist between steps.
    """

    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def start(self):
        """
        Starts the sandbox container and keeps it running.
        """
        try:
            self.container = self.client.containers.run(
                "agent-sandbox",
                command="sleep infinity",  # keeps container alive
                detach=True,
                working_dir="/workspace"
            )
            print("--- Sandbox started ---")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image 'agent-sandbox' not found. Did you build it?")
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Docker is not running or unreachable. Details: {str(e)}")

    def run(self, command: str) -> str:
        """
        Runs a shell command inside the persistent container.
        Returns stdout/stderr as a string.
        """
        if not self.container:
            return "Error: Sandbox is not running. Call start() first."

        try:
            exit_code, output = self.container.exec_run(
                ["sh", "-c", command],  
                workdir="/workspace"
            )
            result = output.decode("utf-8")
            return result if result.strip() else "(no output)"
        except Exception as e:
            return f"Error: {str(e)}"

    def stop(self):
        """
        Stops and removes the container, cleaning up all files.
        """
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                print("--- Sandbox stopped ---")
            except Exception as e:
                print(f"--- Sandbox cleanup error: {str(e)} ---")
            finally:
                self.container = None