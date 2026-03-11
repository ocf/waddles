import asyncio
import subprocess
import os
import textwrap
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

    async def run_python(code: str) -> str:
        """
        Executes arbitrary Python code in a highly secure WebAssembly sandbox.
        """
        # 1. Strip markdown backticks if the LLM added them
        clean_code = code.strip()
        if clean_code.startswith("```"):
            clean_code = "\n".join(clean_code.split("\n")[1:-1])

        # 2. Fix the indentation safely
        clean_code = textwrap.dedent(clean_code.strip("\n"))

        try:
            # Pass the sanitized clean_code instead of the raw code
            output = await asyncio.to_thread(sync_execute_sandboxed, clean_code)
            return output
        except Exception as e:
            return f"Tool execution failed: {e}"

    return FunctionTool.from_defaults(
        async_fn=run_python,
        name="run_python",
        description=(
            "Executes untrusted Python code safely in an isolated environment. "
            "Use this to perform complex math, string manipulation, or logic operations. "
            "You MUST use print() statements to output the final answer, as only console output is returned."
        )
    )
