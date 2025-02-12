"""Dynamic code analysis module."""

import os
import json
import time
import traceback
import coverage
import tracemalloc
import inspect
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from returns.result import Result, Success, Failure

from ..config import Config
from .sandbox import Sandbox
from .exceptions import DynamicAnalysisError


@dataclass
class AnalysisMetrics:
    """Container for dynamic analysis metrics."""

    execution_time: float
    memory_used: int
    exceptions: list
    function_calls: Dict[str, int]
    execution_paths: set
    coverage: Dict[str, float]


class DynamicAnalyzer:
    """Handles dynamic code analysis with sandboxing and instrumentation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.sandbox = Sandbox()

    def analyze(self, code: str, inputs: Optional[Dict[str, Any]] = None) -> Result[AnalysisMetrics, Exception]:
        """Perform dynamic analysis on code.

        Args:
            code: Python code to analyze
            inputs: Optional dictionary of input values

        Returns:
            Result containing analysis metrics

        Raises:
            DynamicAnalysisError: If analysis fails
        """
        try:
            # Add instrumentation for code coverage
            cov = coverage.Coverage()
            cov.start()

            # Add instrumentation code
            instrumented_code = self._instrument_code(code)

            # Run in sandbox
            sandbox_result = self.sandbox.run(instrumented_code, inputs)
            if sandbox_result.is_failure():
                return sandbox_result

            metrics = sandbox_result.unwrap()

            # Add coverage data
            cov.stop()
            coverage_data = cov.get_data()
            metrics["coverage"] = {
                "total_statements": coverage_data.n_statements,
                "covered_statements": coverage_data.n_executed_statements,
                "coverage_percent": coverage_data.percent_covered,
            }

            return Success(
                AnalysisMetrics(
                    execution_time=metrics["execution_time"],
                    memory_used=metrics["memory_used"],
                    exceptions=metrics["exceptions"],
                    function_calls=metrics["function_calls"],
                    execution_paths=set(metrics["execution_paths"]),
                    coverage=metrics["coverage"],
                )
            )

        except Exception as e:
            return Failure(DynamicAnalysisError(f"Dynamic analysis failed: {str(e)}"))

    def _instrument_code(self, code: str) -> str:
        """Add instrumentation to code for analysis.

        Args:
            code: Python code to instrument

        Returns:
            Instrumented code
        """
        return f"""
import sys
import tracemalloc
import time
from collections import defaultdict

# Start tracemalloc for memory profiling
tracemalloc.start()

# Record start time
start_time = time.time()

# Initialize metrics
metrics = {{
    'exceptions': [],
    'memory_snapshots': [],
    'execution_paths': set(),
    'function_calls': defaultdict(int)
}}

# Add exception hook
def exception_hook(exc_type, exc_value, exc_traceback):
    metrics['exceptions'].append({{
        'type': exc_type.__name__,
        'message': str(exc_value),
        'traceback': traceback.format_tb(exc_traceback)
    }})
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_hook

# Add profiling decorator
def profile_function(func):
    def wrapper(*args, **kwargs):
        metrics['function_calls'][func.__name__] += 1
        
        # Take memory snapshot
        snapshot = tracemalloc.take_snapshot()
        metrics['memory_snapshots'].append({{
            'function': func.__name__,
            'memory': snapshot.statistics('filename')
        }})
        
        # Record execution path
        frame = inspect.currentframe()
        metrics['execution_paths'].add(tuple(
            f"{{frame.f_code.co_filename}}:{{frame.f_lineno}}"
            for frame in traceback.extract_stack()
        ))
        
        return func(*args, **kwargs)
    return wrapper

# Add the original code
{code}

# Record end time and metrics
metrics['execution_time'] = time.time() - start_time
metrics['memory_used'] = tracemalloc.get_traced_memory()[1]

# Convert sets to lists for JSON serialization
metrics['execution_paths'] = list(metrics['execution_paths'])

# Store metrics in result
result = metrics
"""
