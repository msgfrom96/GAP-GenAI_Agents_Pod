"""Sandbox module for secure code execution."""

import os
import json
import tempfile
import resource
import subprocess
import logging
from typing import Dict, Any, Optional
from returns.result import Result, Success, Failure

from ..config import Config
from .exceptions import SecurityCheckError


class Sandbox:
    """Handles secure code execution in an isolated environment."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the sandbox.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def run(self, code: str, inputs: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any], Exception]:
        """Run code in a sandboxed environment.

        Args:
            code: Python code to execute
            inputs: Optional dictionary of input values

        Returns:
            Result containing execution metrics and output

        Raises:
            SecurityCheckError: If sandbox limits are exceeded
        """
        try:
            with tempfile.TemporaryDirectory(dir=Config.SANDBOX_DIR) as sandbox_dir:
                # Write code to a file
                code_file = os.path.join(sandbox_dir, "code.py")
                with open(code_file, "w") as f:
                    f.write(code)

                # Write inputs to a file if provided
                if inputs:
                    input_file = os.path.join(sandbox_dir, "inputs.json")
                    with open(input_file, "w") as f:
                        json.dump(inputs, f)

                # Create a wrapper script that sets resource limits
                wrapper_script = self._create_wrapper_script()
                wrapper_file = os.path.join(sandbox_dir, "wrapper.py")
                with open(wrapper_file, "w") as f:
                    f.write(wrapper_script)

                # Run the wrapper script in a subprocess
                result = subprocess.run(
                    [Config.SANDBOX_PYTHON, wrapper_file],
                    cwd=sandbox_dir,
                    capture_output=True,
                    timeout=Config.SANDBOX_TIMEOUT + 5,  # Add buffer for setup time
                )

                # Check for errors
                if result.returncode != 0:
                    return Failure(SecurityCheckError(f"Code execution failed: {result.stderr.decode()}"))

                # Load and return metrics
                try:
                    with open(os.path.join(sandbox_dir, "metrics.json")) as f:
                        metrics = json.load(f)
                    return Success(metrics)
                except Exception as e:
                    return Failure(SecurityCheckError(f"Failed to load metrics: {str(e)}"))

        except Exception as e:
            return Failure(SecurityCheckError(f"Sandbox execution failed: {str(e)}"))

    def _create_wrapper_script(self) -> str:
        """Create the Python wrapper script for sandboxed execution.

        Returns:
            String containing the wrapper script code
        """
        return f"""
import resource
import sys
import json
import time
import psutil

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, ({Config.SANDBOX_TIMEOUT}, {Config.SANDBOX_TIMEOUT}))
resource.setrlimit(resource.RLIMIT_AS, ({Config.SANDBOX_MEMORY_LIMIT * 1024 * 1024}, {Config.SANDBOX_MEMORY_LIMIT * 1024 * 1024}))

# Record start time and resources
start_time = time.time()
start_memory = psutil.Process().memory_info().rss

# Execute the code
with open('code.py') as f:
    code = compile(f.read(), 'code.py', 'exec')
    
# Create a clean namespace
namespace = {{}}

# Load inputs if they exist
try:
    with open('inputs.json') as f:
        namespace['inputs'] = json.load(f)
except FileNotFoundError:
    namespace['inputs'] = None

# Execute the code
exec(code, namespace)

# Record end time and resources
end_time = time.time()
end_memory = psutil.Process().memory_info().rss

# Calculate metrics
metrics = {{
    'execution_time': end_time - start_time,
    'memory_used': end_memory - start_memory,
    'output': str(namespace.get('result', None))
}}

# Write metrics to file
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
"""
