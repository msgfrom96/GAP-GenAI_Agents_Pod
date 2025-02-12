"""Test coverage analysis module."""

import os
import json
import logging
import coverage
import pytest
import re
import ast
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from returns.result import Result, Success, Failure

from ..config import Config
from .exceptions import TestCoverageError


@dataclass
class CoverageMetrics:
    """Container for coverage metrics."""

    total_coverage: float
    unit_coverage: float
    integration_coverage: float
    uncovered_lines: Dict[str, List[int]]
    critical_paths: Dict[str, float]


class TestCoverageVerifier:
    """Handles test coverage analysis and verification."""

    def __init__(self, github_manager, logger: Optional[logging.Logger] = None):
        """Initialize the coverage verifier.

        Args:
            github_manager: GitHub manager for issue creation
            logger: Optional logger instance
        """
        self.github_manager = github_manager
        self.logger = logger or logging.getLogger(__name__)

    def analyze_coverage(self) -> Result[CoverageMetrics, Exception]:
        """Analyze test coverage.

        Returns:
            Result containing coverage metrics

        Raises:
            TestCoverageError: If analysis fails
        """
        try:
            metrics = CoverageMetrics(
                total_coverage=0.0, unit_coverage=0.0, integration_coverage=0.0, uncovered_lines={}, critical_paths={}
            )

            # Run tests with coverage
            cov = coverage.Coverage()
            cov.start()

            # Run unit tests
            unit_result = pytest.main(["-m", "unit", "--cov"])
            if unit_result == 0:
                unit_data = cov.get_data()
                metrics.unit_coverage = unit_data.percent_covered

            # Run integration tests
            integration_result = pytest.main(["-m", "integration", "--cov"])
            if integration_result == 0:
                integration_data = cov.get_data()
                metrics.integration_coverage = integration_data.percent_covered

            cov.stop()

            # Get total coverage
            total_data = cov.get_data()
            metrics.total_coverage = total_data.percent_covered

            # Find uncovered lines
            for filename in total_data.measured_files():
                _, executable, missing, _ = cov.analysis(filename)
                if missing:
                    metrics.uncovered_lines[filename] = missing

            # Analyze critical paths
            metrics.critical_paths = self._analyze_critical_paths(total_data)

            # Save results
            Config.save_state(
                Config.COVERAGE_HISTORY,
                {
                    "total": metrics.total_coverage,
                    "unit": metrics.unit_coverage,
                    "integration": metrics.integration_coverage,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Check coverage thresholds
            self._check_coverage_thresholds(metrics)

            return Success(metrics)

        except Exception as e:
            self.logger.exception("Coverage analysis failed")
            return Failure(TestCoverageError(f"Coverage analysis failed: {str(e)}"))

    def _analyze_critical_paths(self, coverage_data) -> Dict[str, float]:
        """Analyze coverage of critical code paths.

        Args:
            coverage_data: Coverage data object

        Returns:
            Dictionary mapping critical paths to coverage percentage
        """
        critical_paths = {}
        critical_patterns = [r"def handle_.*", r"def process_.*", r"def validate_.*", r"class.*Error", r"def _check_.*"]

        for filename in coverage_data.measured_files():
            with open(filename) as f:
                content = f.read()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function matches critical patterns
                        if any(re.match(pattern, node.name) for pattern in critical_patterns):
                            # Get coverage for this function
                            start_line = node.lineno
                            end_line = max(node.body[-1].lineno, node.lineno)
                            lines = set(range(start_line, end_line + 1))

                            # Get missing lines
                            _, executable, missing, _ = coverage_data.analysis(filename)
                            missing_set = set(missing)

                            # Calculate coverage
                            covered_lines = lines - missing_set
                            if lines:
                                coverage_percent = len(covered_lines) / len(lines) * 100
                                critical_paths[f"{filename}:{node.name}"] = coverage_percent

        return critical_paths

    def _check_coverage_thresholds(self, metrics: CoverageMetrics) -> None:
        """Check if coverage meets thresholds.

        Args:
            metrics: Coverage metrics to check
        """
        issues = []

        # Check total coverage
        if metrics.total_coverage < Config.COVERAGE_THRESHOLDS["min_total"]:
            issues.append(
                {
                    "title": "Insufficient Total Test Coverage",
                    "body": f"Total coverage ({metrics.total_coverage:.1f}%) is below minimum threshold ({Config.COVERAGE_THRESHOLDS['min_total']}%)",
                    "labels": ["test-coverage", "total-coverage"],
                }
            )

        # Check unit test coverage
        if metrics.unit_coverage < Config.COVERAGE_THRESHOLDS["min_unit"]:
            issues.append(
                {
                    "title": "Insufficient Unit Test Coverage",
                    "body": f"Unit test coverage ({metrics.unit_coverage:.1f}%) is below minimum threshold ({Config.COVERAGE_THRESHOLDS['min_unit']}%)",
                    "labels": ["test-coverage", "unit-tests"],
                }
            )

        # Check integration test coverage
        if metrics.integration_coverage < Config.COVERAGE_THRESHOLDS["min_integration"]:
            issues.append(
                {
                    "title": "Insufficient Integration Test Coverage",
                    "body": f"Integration test coverage ({metrics.integration_coverage:.1f}%) is below minimum threshold ({Config.COVERAGE_THRESHOLDS['min_integration']}%)",
                    "labels": ["test-coverage", "integration-tests"],
                }
            )

        # Check critical paths
        for path, coverage in metrics.critical_paths.items():
            if coverage < Config.COVERAGE_THRESHOLDS["critical_paths"]:
                issues.append(
                    {
                        "title": "Critical Path Undercovered",
                        "body": f"Critical path {path} has insufficient coverage ({coverage:.1f}%)",
                        "labels": ["test-coverage", "critical-path"],
                    }
                )

        # Create GitHub issues
        for issue in issues:
            self.github_manager.create_issue(**issue)

    def get_uncovered_code(self) -> Result[Dict[str, List[str]], Exception]:
        """Get uncovered code snippets.

        Returns:
            Result containing uncovered code by file

        Raises:
            TestCoverageError: If retrieval fails
        """
        try:
            uncovered_code = {}

            # Run coverage
            cov = coverage.Coverage()
            cov.start()
            pytest.main(["--cov"])
            cov.stop()

            # Get uncovered lines
            for filename in cov.get_data().measured_files():
                _, executable, missing, _ = cov.analysis(filename)
                if missing:
                    with open(filename) as f:
                        lines = f.readlines()
                        uncovered_code[filename] = [
                            lines[line_no - 1].strip() for line_no in missing if 0 <= line_no - 1 < len(lines)
                        ]

            return Success(uncovered_code)

        except Exception as e:
            self.logger.exception("Failed to get uncovered code")
            return Failure(TestCoverageError(f"Failed to get uncovered code: {str(e)}"))

    def get_coverage_history(self) -> Result[List[Dict[str, Any]], Exception]:
        """Get historical coverage data.

        Returns:
            Result containing coverage history

        Raises:
            TestCoverageError: If retrieval fails
        """
        try:
            history = Config.load_state(Config.COVERAGE_HISTORY)
            return Success(history if history else [])

        except Exception as e:
            self.logger.exception("Failed to get coverage history")
            return Failure(TestCoverageError(f"Failed to get coverage history: {str(e)}"))

    def find_untested_changes(self, commit_sha: str) -> Result[Set[str], Exception]:
        """Find changed files without test coverage.

        Args:
            commit_sha: Commit SHA to check

        Returns:
            Result containing set of untested files

        Raises:
            TestCoverageError: If check fails
        """
        try:
            # Get changed files
            changed_files = self.github_manager.get_commit(commit_sha).files
            changed_python_files = {f.filename for f in changed_files if f.filename.endswith(".py")}

            # Get test files
            test_files = {f for f in changed_python_files if f.startswith("tests/") or f.endswith("_test.py")}

            # Find source files without tests
            source_files = changed_python_files - test_files
            untested_files = set()

            for source_file in source_files:
                # Check for corresponding test file
                test_file = f"tests/test_{os.path.basename(source_file)}"
                if test_file not in test_files:
                    untested_files.add(source_file)

            return Success(untested_files)

        except Exception as e:
            self.logger.exception("Failed to find untested changes")
            return Failure(TestCoverageError(f"Failed to find untested changes: {str(e)}"))

    def suggest_test_improvements(self, metrics: CoverageMetrics) -> Result[List[str], Exception]:
        """Suggest improvements for test coverage.

        Args:
            metrics: Current coverage metrics

        Returns:
            Result containing list of suggestions

        Raises:
            TestCoverageError: If analysis fails
        """
        try:
            suggestions = []

            # Check total coverage
            if metrics.total_coverage < Config.COVERAGE_THRESHOLDS["min_total"]:
                suggestions.append(
                    f"Increase total coverage from {metrics.total_coverage:.1f}% to "
                    f"at least {Config.COVERAGE_THRESHOLDS['min_total']}%"
                )

            # Check critical paths
            for path, coverage in metrics.critical_paths.items():
                if coverage < Config.COVERAGE_THRESHOLDS["critical_paths"]:
                    suggestions.append(f"Add tests for critical path {path} " f"(current coverage: {coverage:.1f}%)")

            # Check uncovered lines
            for file, lines in metrics.uncovered_lines.items():
                if len(lines) > 10:  # Suggest for files with significant uncovered code
                    suggestions.append(f"Add tests for {file} " f"({len(lines)} uncovered lines)")

            return Success(suggestions)

        except Exception as e:
            self.logger.exception("Failed to generate test suggestions")
            return Failure(TestCoverageError(f"Failed to generate test suggestions: {str(e)}"))
