import asyncio
import subprocess
import os
from llama_index.core.tools import FunctionTool

def create_python_run_tool() -> FunctionTool:
    """Create a tool to execute arbitrary Python code securely in a Wasm sandbox."""

    def sync_execute_sandboxed(code_string: str) -> str:
        """Synchronous execution of the Node.js Pyodide sandbox."""

        # We use the modern ESM import style that worked perfectly on Node v25
        js_runner = """
        import fs from 'fs';
        import { loadPyodide } from 'pyodide';

        try {
            const pyodide = await loadPyodide();
            const code = fs.readFileSync(0, 'utf-8');

            // Execute the sandbox and capture the final expression's result
            const result = await pyodide.runPythonAsync(code);

            // If the code evaluated to something, log it so stdout captures it
            if (result !== undefined) {
                console.log(result);
            }
        } catch (err) {
            console.error("[Sandbox Execution Error]:\\n", err);
            process.exit(1);
        }
        """

        # Strict check to ensure the agent doesn't get confused by missing dependencies
        if not os.path.exists(os.path.join(os.getcwd(), "node_modules", "pyodide")):
            return "Execution Error: 'node_modules/pyodide' not found. Ensure the host environment has run 'npm install pyodide'."

        try:
            result = subprocess.run(
                ['node', '--input-type=module', '-e', js_runner],
                input=code_string,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=15 # Important for LLMs: Prevent infinite loops from hanging the agent
            )

            # Combine stdout and stderr into a single string for the LLM to read
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"

            if not output.strip():
                return "Execution completed successfully, but there was no console output."

            return output.strip()

        except subprocess.TimeoutExpired:
            return "Execution Error: Code timed out after 15 seconds. You may have written an infinite loop."
        except FileNotFoundError:
            return "Execution Error: Node.js is not installed or not in the system PATH."
        except Exception as e:
            return f"Execution Error: Tool failed unexpectedly: {e}"

    async def python_run(code: str) -> str:
        """
        Executes arbitrary Python code in a highly secure WebAssembly sandbox.

        Args:
            code: The Python script to execute. Use print() statements to output results.
        """
        try:
            # Offload the blocking subprocess execution to a thread
            output = await asyncio.to_thread(sync_execute_sandboxed, code)
            return output
        except Exception as e:
            return f"Tool execution failed: {e}"

    return FunctionTool.from_defaults(
        async_fn=python_run,
        name="python_run",
        description=(
            "Executes untrusted Python code safely in an isolated environment. "
            "Use this to perform complex math, string manipulation, or logic operations. "
            "You MUST use print() statements to output the final answer, as only console output is returned."
        )
    )
