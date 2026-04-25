import docker
import time


class Sandbox:
    

    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def start(self):
        
        try:
            self.container = self.client.containers.run(
                "agent-sandbox",
                command="sleep infinity",
                detach=True,
                working_dir="/workspace"
            )
            
            time.sleep(1)
            print("--- Sandbox started ---")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image 'agent-sandbox' not found. Did you build it?")
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Docker is not running or unreachable. Details: {str(e)}")

    def run(self, command: str) -> str:
        
        if not self.container:
            return "Error: Sandbox is not running. Call start() first."

        try:
            exit_code, output = self.container.exec_run(
                ["sh", "-c", f"cd /workspace && {command}"],
                workdir="/workspace"
            )
            result = output.decode("utf-8")

            
            if any(op in command for op in ["cat >", ">>"]):
                time.sleep(0.5)

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