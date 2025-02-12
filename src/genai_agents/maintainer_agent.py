import os
import logging
import time
import asyncio
import inspect
import re
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import pytest
import ast
from collections import defaultdict
import json
from typing import Optional, List, Dict, Any
import hashlib
from dataclasses import dataclass, asdict
import yaml
from cryptography.fernet import Fernet
from functools import wraps
from unittest.mock import Mock
import subprocess
import psutil
from returns.result import Result, Success, Failure
import config  # Import the config module
from tenacity import retry, stop_after_attempt, wait_exponential
import astunparse  # For converting AST back to code (for debugging)
import aiohttp

from .config import Config
from .openai import OpenAIService
from .google import GoogleAIService
from .github import GitHubClient, GitHubConfigError
from .security import SecurityChecker, SecurityCheckError
from .documentation import DocumentationAnalyzer, DocumentationState, DocumentationAnalysisResult
from .monitoring import ModelMonitor
from .pricing import PricingUpdater
from .code_reviewer import CodeReviewer
from .dependency_analyzer import DependencyAnalyzer
from .dynamic_analyzer import DynamicAnalyzer
from .test_coverage import TestCoverageAnalyzer
from .sandbox import CodeSandbox
from .state_manager import StateManager
from .exceptions import (
    MaintainerError,
    SecurityCheckError,
    DocumentationError,
    PricingError,
    ModelMonitoringError,
    CodeReviewError,
    DependencyError,
    TestCoverageError,
    StateError,
)
from .maintainer.document_store import DocumentStore
from .maintainer.code_reviewer import CodeReviewer
from .maintainer.dependency_analyzer import DependencyAnalyzer
from .maintainer.documentation import DocumentationAnalyzer
from .maintainer.dynamic_analyzer import DynamicAnalyzer
from .maintainer.test_coverage import TestCoverageVerifier
from .maintainer.model_monitor import ModelMonitor
from .maintainer.pricing import PricingAnalyzer
from .maintainer.github import GitHubManager
from .maintainer.state_manager import StateManager, State

# Define Prometheus metrics
from prometheus_client import Summary, Gauge

MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy", ["version"])
MODEL_LATENCY = Summary("model_latency", "Model latency", ["version"])
MODEL_THROUGHPUT = Gauge("model_throughput", "Model throughput", ["version"])


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not getattr(args[0], "is_authenticated", False):
            raise PermissionError("Login required")
        return f(*args, **kwargs)

    return decorated_function


class MaintainerAgent:
    """AI-powered code maintenance agent with MLOps capabilities"""

    def __init__(self, google_ai_system, openai_system, test_mode: bool = False):
        """Initialize the maintainer agent.

        Args:
            google_ai_system: Google AI service instance
            openai_system: OpenAI service instance
            test_mode: Whether to use mock services

        Raises:
            GitHubConfigError: If GitHub configuration is invalid
            ValueError: If required services are not provided
            StateError: If state loading fails
        """
        if not google_ai_system or not openai_system:
            raise ValueError("Both AI systems must be provided")

        self.test_mode = test_mode or Config.TEST_MODE

        # Store AI systems
        self.google_system = google_ai_system
        self.openai_system = openai_system

        # Setup GitHub client
        if self.test_mode:
            self._setup_mocks()
        else:
            try:
                self.github_client = GitHubClient(github_client)
                self.repo = self.github_client.repo
                self.repo_name = self.github_client.repo_name
            except Exception as e:
                raise GitHubConfigError(f"Failed to initialize GitHub client: {str(e)}")

        # Initialize state
        self.last_check = None
        self.deprecated_functions = set()
        self.new_features_queue = []
        self.model_performance = {}  # Track accuracy/latency
        self.retraining_thresholds = Config.MODEL_THRESHOLDS
        self.versions = {"google_ai": "1.8.2", "openai": "1.12.0", "tensorflow": "2.16.1"}
        self.active_model_version = self.versions["google_ai"]
        self.is_authenticated = True if self.test_mode else False

        # Configure logging
        self.logger = logging.getLogger("MaintainerAgent")
        self.logger.setLevel(logging.INFO)

        # Initialize documentation analyzer
        self.doc_analyzer = DocumentationAnalyzer(self.openai_system, self.logger, self.test_mode)
        self.doc_states: Dict[str, DocumentationState] = {}
        self._load_doc_states()

        # Initialize security checker
        self.security_checker = SecurityChecker(self.logger, self.github_client)

        # Initialize model monitor
        self.model_monitor = ModelMonitor(self.logger)

        # Initialize pricing updater
        self.pricing_updater = PricingUpdater(self.logger, self.github_client)

        # Initialize code reviewer
        self.code_reviewer = CodeReviewer(
            self.logger, self.github_client, self.deprecated_functions, self._submit_github_issue
        )

        # Initialize dependency analyzer
        self.dependency_analyzer = DependencyAnalyzer(self.logger)

        # Initialize dynamic analyzer
        self.dynamic_analyzer = DynamicAnalyzer(self.logger, self._run_in_sandbox)

        # Initialize test coverage analyzer
        self.test_coverage_analyzer = TestCoverageAnalyzer(self.logger, self._submit_github_issue)

        # Initialize code sandbox
        self.code_sandbox = CodeSandbox(self.logger)

        # Initialize state manager
        self.state_manager = StateManager(self.logger)

        # Initialize encryption key
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Initialize pricing data
        self.pricing_data: Dict[str, Dict[str, Any]] = {}
        self._load_pricing_data()

        self.github_repo = config.GITHUB_REPO  # Use the config variable

        # Initialize state and managers
        self.state_manager = StateManager(logger=self.logger)
        self.github_manager = GitHubManager(test_mode=test_mode, logger=self.logger)

        # Initialize document store
        self.doc_store = DocumentStore(logger=self.logger)

        # Load initial state
        state_result = self.state_manager.load_state()
        if state_result.is_failure():
            self.logger.error(f"Failed to load state: {state_result.failure()}")
            raise StateError("Failed to load initial state")
        self.state = state_result.unwrap()

        # Initialize HTTP session for async requests
        self.session = aiohttp.ClientSession()

    def _setup_mocks(self):
        """Set up mock objects for testing"""
        # Mock GitHub client
        self.github_client = Mock()
        mock_repo = Mock()
        mock_repo.get_commits = Mock(return_value=[])
        mock_repo.create_issue = Mock(return_value=Mock(number=1))
        mock_repo.get_issues = Mock(return_value=[])
        self.github_client.get_repo = Mock(return_value=mock_repo)

        # Mock documentation states
        self.doc_states = {
            "Google": DocumentationState(
                content_hash="mock_hash",
                last_updated=datetime.now(),
                deprecated_functions=["mock_deprecated_func"],
                new_features=["mock_new_feature"],
                security_updates=["mock_security_update"],
                breaking_changes=["mock_breaking_change"],
            ),
            "OpenAI": DocumentationState(
                content_hash="mock_hash",
                last_updated=datetime.now(),
                deprecated_functions=["mock_deprecated_func"],
                new_features=["mock_new_feature"],
                security_updates=["mock_security_update"],
                breaking_changes=["mock_breaking_change"],
            ),
        }

        # Mock scheduler
        self.scheduler = Mock()
        self.scheduler.add_job = Mock()
        self.scheduler.start = Mock()
        self.scheduler.shutdown = Mock()

        # Mock test results
        self.model_performance = {"mock-model-v1": {"accuracy": 0.95, "latency": 1.0, "throughput": 1000}}

        # Mock pricing data
        self.pricing_data = {
            "OpenAI": {"gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}},
            "Google": {"gemini-1.0-pro": {"prompt": 0.00025, "completion": 0.0005}},
        }

    def _load_doc_states(self):
        """Load previous documentation states from disk"""
        try:
            with open("doc_states.yaml", "r") as f:
                states = yaml.safe_load(f)
                for api, state in states.items():
                    self.doc_states[api] = DocumentationState(**state)
        except FileNotFoundError:
            self.logger.info("No previous documentation states found")

    def _save_doc_states(self):
        """Save current documentation states to disk"""
        states = {api: asdict(state) for api, state in self.doc_states.items()}
        with open("doc_states.yaml", "w") as f:
            yaml.dump(states, f)

    def _load_pricing_data(self):
        """Load previous pricing data from disk"""
        try:
            with open("pricing_data.json", "r") as f:
                self.pricing_data = json.load(f)
        except FileNotFoundError:
            self.logger.info("No previous pricing data found")
            self.pricing_data = {}

    def _save_pricing_data(self):
        """Save current pricing data to disk"""
        with open("pricing_data.json", "w") as f:
            json.dump(self.pricing_data, f)

    async def _fetch_latest_documentation(self, api_name: str) -> str:
        """Retrieve latest API documentation"""
        return await self.doc_analyzer._fetch_latest_documentation(api_name)

    async def _fetch_pricing_data(self, api_name: str) -> Dict[str, Dict[str, Any]]:
        """Fetch the latest pricing information from the API provider's website."""
        return await self.pricing_updater._fetch_pricing_data(api_name)

    def _extract_openai_pricing(self, html_content: str) -> Dict[str, Dict[str, float]]:
        """Extract OpenAI pricing data from HTML content."""
        return self.pricing_updater._extract_openai_pricing(html_content)

    def _extract_google_pricing(self, html_content: str) -> Dict[str, Dict[str, float]]:
        """Extract Google AI pricing data from HTML content."""
        return self.pricing_updater._extract_google_pricing(html_content)

    def _compare_pricing_data(self, api_name: str, new_data: Dict[str, Dict[str, Any]]):
        """Compare new pricing data with existing data and identify significant changes."""
        self.pricing_updater._compare_pricing_data(api_name, new_data, self.pricing_data)

    def _find_google_deprecations(self, doc_content: str) -> list:
        """Identify deprecated Google AI functions"""
        return self.doc_analyzer._find_google_deprecations(doc_content)

    def _find_openai_deprecations(self, doc_content: str) -> list:
        """Identify deprecated OpenAI functions"""
        return self.doc_analyzer._find_openai_deprecations(doc_content)

    def _find_google_new_features(self, doc_content: str) -> list:
        """Detect new Google AI features"""
        return self.doc_analyzer._find_google_new_features(doc_content)

    def _find_openai_new_features(self, doc_content: str) -> list:
        """Detect new OpenAI features"""
        return self.doc_analyzer._find_openai_new_features(doc_content)

    def _check_code_against_docs(self):
        """Compare current code implementation with documentation"""
        google_changes = self._parse_documentation("Google")
        openai_changes = self._parse_documentation("OpenAI")

        # Check Google implementation
        current_google_funcs = set(name for name, _ in inspect.getmembers(self.google_system, inspect.ismethod))
        self.deprecated_functions.update(func for func in google_changes.deprecated if func in current_google_funcs)

        # Check OpenAI implementation
        current_openai_funcs = set(name for name, _ in inspect.getmembers(self.openai_system, inspect.ismethod))
        self.deprecated_functions.update(func for func in openai_changes.deprecated if func in current_openai_funcs)

        # Queue new features
        self.new_features_queue.extend(google_changes.new_features)
        self.new_features_queue.extend(openai_changes.new_features)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _submit_github_issue(self, title: str, body: str) -> Result[bool, Exception]:
        """Create GitHub issue for required changes.

        Args:
            title: Issue title
            body: Issue description

        Returns:
            Result containing True if issue was created, False if it already exists

        Raises:
            ValueError: If title or body is invalid
            GitHubConfigError: If GitHub configuration is invalid
        """
        return self.github_client.submit_github_issue(title, body)

    def _validate_code_changes(self, new_code: str) -> bool:
        """Validate generated code with AST parsing and test suite"""
        try:
            ast.parse(new_code)
            test_result = pytest.main(["--tb=line", "-k", "test_code_generation"])
            return test_result == 0
        except SyntaxError as e:
            self.logger.error(f"Invalid syntax: {str(e)}")
            return False

    async def _test_functionality(self):
        """Run comprehensive test suite"""
        test_results = {
            "unit_tests": pytest.main(["-m", "unit"]),
            "integration_tests": pytest.main(["-m", "integration"]),
            "load_tests": pytest.main(["-m", "load"]),
        }

        if any(result != 0 for result in test_results.values()):
            raise RuntimeError("Test failures detected")

        self.logger.info("All tests passed successfully")
        return True

    def _update_documentation(self):
        """Update project documentation with new features"""
        try:
            docs_dir = "docs"
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)

            docs_file = os.path.join(docs_dir, "api_changes.md")
            with open(docs_file, "a") as f:
                f.write(f"\n# New Features\n# {datetime.now().isoformat()}\n")
                for feature in self.new_features_queue:
                    f.write(f"# - {feature}\n")
        except Exception as e:
            self.logger.error(f"Documentation update failed: {str(e)}")

    def _perform_code_review(self) -> Result[None, Exception]:
        """Analyze code for deprecated patterns and potential issues using diffs"""
        last_analyzed_commit_sha = self._get_last_analyzed_commit()

        try:
            # Get the diff between the last analyzed commit and the latest commit
            diff_data = self.github_client.get_diff(last_analyzed_commit_sha)
            if not diff_data:
                self.logger.info("No new commits to analyze or no diff found.")
                return Success(None)

            # Review the diff using the CodeReviewer
            review_result = self.code_reviewer.review_diff(diff_data)
            if review_result.is_failure():
                self.logger.error(f"Code review failed: {review_result.failure()}")
                return Failure(review_result.failure())

            # Update the last analyzed commit SHA
            latest_commit_sha = self.github_client.get_latest_commit_sha()
            if not latest_commit_sha:
                self.logger.warning("Could not retrieve latest commit SHA after review.")
                return Success(None)

            self._set_last_analyzed_commit(latest_commit_sha)
            return Success(None)

        except Exception as e:
            self.logger.exception(f"Error during code review: {e}")
            return Failure(CodeReviewError(f"Error during code review: {str(e)}"))

    def _check_subprocess_usage(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for unsafe subprocess usage."""
        return self.security_checker._check_subprocess_usage(tree)

    def _check_deserialization(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for unsafe deserialization."""
        return self.security_checker._check_deserialization(tree)

    def _check_input_validation(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for missing input validation."""
        return self.security_checker._check_input_validation(tree)

    def _check_error_handling(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for proper error handling."""
        return self.security_checker._check_error_handling(tree)

    def _check_hardcoded_secrets(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for hardcoded secrets."""
        return self.security_checker._check_hardcoded_secrets(tree)

    @staticmethod
    def _schedule_weekly_maintenance():
        """Configure weekly maintenance schedule"""
        scheduler = BackgroundScheduler()
        scheduler.add_job(MaintainerAgent.run_maintenance_cycle, "interval", weeks=1, next_run_time=datetime.now())
        scheduler.start()

    async def run(self) -> Result[bool, Exception]:
        """Run the maintenance cycle.

        Returns:
            Result indicating success

        Raises:
            MaintainerError: If maintenance fails
        """
        try:
            self.logger.info("Starting maintenance cycle")
            start_time = datetime.now()

            # Phase 1: Documentation Analysis
            self.logger.info("Phase 1: Documentation Analysis")
            doc_result = await self._analyze_documentation()
            if doc_result.is_failure():
                return doc_result

            # Phase 2: Code Analysis
            self.logger.info("Phase 2: Code Analysis")
            code_result = await self._analyze_code()
            if code_result.is_failure():
                return code_result

            # Phase 3: Dependency Analysis
            self.logger.info("Phase 3: Dependency Analysis")
            dep_result = await self._analyze_dependencies()
            if dep_result.is_failure():
                return dep_result

            # Phase 4: Test Coverage Analysis
            self.logger.info("Phase 4: Test Coverage Analysis")
            coverage_result = await self._analyze_test_coverage()
            if coverage_result.is_failure():
                return coverage_result

            # Phase 5: Model Performance Analysis
            self.logger.info("Phase 5: Model Performance Analysis")
            perf_result = await self._analyze_model_performance()
            if perf_result.is_failure():
                return perf_result

            # Phase 6: Pricing Analysis
            self.logger.info("Phase 6: Pricing Analysis")
            pricing_result = await self._analyze_pricing()
            if pricing_result.is_failure():
                return pricing_result

            # Update state
            self.state.last_run = start_time.isoformat()
            self.state.duration = (datetime.now() - start_time).total_seconds()
            self.state.phases_completed = 6

            # Save final state
            save_result = await self.state_manager.save_state(self.state)
            if save_result.is_failure():
                return save_result

            self.logger.info(f"Maintenance cycle completed in {self.state.duration:.1f}s")
            return Success(True)

        except Exception as e:
            self.logger.exception("Maintenance cycle failed")
            return Failure(MaintainerError(f"Maintenance cycle failed: {str(e)}"))

    async def _analyze_documentation(self) -> Result[bool, Exception]:
        """Analyze API documentation."""
        try:
            # Analyze both APIs concurrently
            results = await asyncio.gather(
                self.documentation_analyzer.analyze_documentation("OpenAI"),
                self.documentation_analyzer.analyze_documentation("Google"),
                return_exceptions=True,
            )

            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    return Failure(DocumentationError(str(result)))

            return Success(True)

        except Exception as e:
            self.logger.exception("Documentation analysis failed")
            return Failure(DocumentationError(f"Documentation analysis failed: {str(e)}"))

    async def _analyze_code(self) -> Result[bool, Exception]:
        """Analyze code changes."""
        try:
            # Get latest commit from GitHub
            commit_result = await self.github_manager.get_latest_commit()
            if commit_result.is_failure():
                return commit_result

            latest_commit = commit_result.unwrap()

            # Skip if no new changes
            if latest_commit.sha == self.state.last_analyzed_commit:
                self.logger.info("No new code changes to analyze")
                return Success(True)

            # Get changed files
            files_result = await self.github_manager.get_changed_files(latest_commit.sha)
            if files_result.is_failure():
                return files_result

            changed_files = files_result.unwrap()

            # Analyze each file concurrently
            tasks = []
            for file_info in changed_files:
                if file_info.filename.endswith(".py"):
                    tasks.extend(
                        [
                            self.code_reviewer.review_code(file_info.patch, file_info.filename),
                            self.dynamic_analyzer.analyze(file_info.patch),
                        ]
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    return Failure(CodeReviewError(str(result)))

            # Update last analyzed commit
            self.state.last_analyzed_commit = latest_commit.sha

            return Success(True)

        except Exception as e:
            self.logger.exception("Code analysis failed")
            return Failure(CodeReviewError(f"Code analysis failed: {str(e)}"))

    async def _analyze_dependencies(self) -> Result[bool, Exception]:
        """Analyze project dependencies."""
        try:
            # Run dependency analysis
            dep_result = await self.dependency_analyzer.analyze_dependencies()
            if dep_result.is_failure():
                return dep_result

            # Check for updates if needed
            if dep_result.unwrap()["outdated"]:
                update_result = await self.dependency_analyzer.update_dependencies()
                if update_result.is_failure():
                    return update_result

            return Success(True)

        except Exception as e:
            self.logger.exception("Dependency analysis failed")
            return Failure(DependencyError(f"Dependency analysis failed: {str(e)}"))

    async def _analyze_test_coverage(self) -> Result[bool, Exception]:
        """Analyze test coverage."""
        try:
            # Run coverage analysis
            coverage_result = await self.test_coverage.analyze_coverage()
            if coverage_result.is_failure():
                return coverage_result

            metrics = coverage_result.unwrap()

            # Get suggestions for improvement
            suggestions_result = await self.test_coverage.suggest_test_improvements(metrics)
            if suggestions_result.is_failure():
                return suggestions_result

            return Success(True)

        except Exception as e:
            self.logger.exception("Test coverage analysis failed")
            return Failure(TestCoverageError(f"Test coverage analysis failed: {str(e)}"))

    async def _analyze_model_performance(self) -> Result[bool, Exception]:
        """Analyze model performance."""
        try:
            # Get test data
            test_data = await self._get_test_data()

            # Test both models concurrently
            results = await asyncio.gather(
                self.model_monitor.run_performance_test(self.openai, test_data),
                self.model_monitor.run_performance_test(self.google_system, test_data),
                return_exceptions=True,
            )

            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    return Failure(ModelMonitoringError(str(result)))

            return Success(True)

        except Exception as e:
            self.logger.exception("Model performance analysis failed")
            return Failure(ModelMonitoringError(f"Model performance analysis failed: {str(e)}"))

    async def _analyze_pricing(self) -> Result[bool, Exception]:
        """Analyze API pricing changes."""
        try:
            # Check both APIs concurrently
            results = await asyncio.gather(
                self.pricing_analyzer.update_pricing_data("OpenAI"),
                self.pricing_analyzer.update_pricing_data("Google"),
                return_exceptions=True,
            )

            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    return Failure(PricingError(str(result)))

            return Success(True)

        except Exception as e:
            self.logger.exception("Pricing analysis failed")
            return Failure(PricingError(f"Pricing analysis failed: {str(e)}"))

    async def _get_test_data(self) -> List[Dict[str, Any]]:
        """Get test data for model evaluation."""
        # This would typically load from a test dataset
        # For now, return a simple test set
        return [
            {"input": "What is machine learning?", "expected": "Machine learning is..."},
            {"input": "Explain neural networks", "expected": "Neural networks are..."},
            {"input": "How does GPT work?", "expected": "GPT (Generative Pre-trained Transformer)..."},
        ]

    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Result[str, Exception]:
        """Add a document to the store.

        Args:
            content: Document content
            metadata: Optional document metadata

        Returns:
            Result containing document ID

        Raises:
            ValueError: If content is invalid
        """
        try:
            # Generate embedding
            if self.test_mode:
                embedding = [0.1] * 1536  # Mock embedding
            else:
                embedding = await self.openai_system.generate_embedding(content)

            # Add to document store
            return await self.doc_store.add_document(content, embedding, metadata)

        except Exception as e:
            self.logger.exception("Failed to add document")
            return Failure(StateError(f"Failed to add document: {str(e)}"))

    async def start(self):
        """Start the maintenance agent."""
        async with self:
            return await self.run()

    async def stop(self):
        """Stop the maintenance agent."""
        await self.cleanup()

    def _get_python_files(self) -> List[str]:
        """Get all Python files in the repository."""
        python_files = []
        for root, _, files in os.walk("."):
            if "venv" in root or ".git" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    async def _update_pricing_information(self):
        """Update pricing information for OpenAI and Google AI models."""
        await self.pricing_updater.update_pricing_information(self.pricing_data)

    def _monitor_model_performance(self):
        """Track model metrics across versions"""
        current_stats = {
            "accuracy": self._run_accuracy_test(),
            "latency": self._measure_inference_speed(),
            "throughput": self._stress_test(),
        }
        self.model_performance[self.active_model_version] = current_stats

        # Expose metrics to Prometheus
        MODEL_ACCURACY.labels(version=self.active_model_version).set(current_stats["accuracy"])
        MODEL_LATENCY.labels(version=self.active_model_version).observe(current_stats["latency"])
        MODEL_THROUGHPUT.labels(version=self.active_model_version).set(current_stats["throughput"])

    def _check_retraining_needs(self):
        """Evaluate if retraining is needed"""
        current = self.model_performance.get(self.active_model_version, {})
        if (
            current.get("accuracy", 1.0) < self.retraining_thresholds["accuracy"]
            or current.get("latency", 0.0) > self.retraining_thresholds["latency"]
        ):
            self._submit_github_issue("Model Retraining Recommended", f"Performance degradation detected: {current}")
            # Trigger rollback to the previous version
            previous_version = self._get_previous_model_version()
            if previous_version:
                self._handle_rollback(previous_version)
            else:
                self.logger.warning("No previous model version found. Cannot rollback.")

    def start(self):
        """Start scheduled maintenance"""
        self.scheduler.add_job(self.run_maintenance_cycle, "interval", weeks=1, next_run_time=datetime.now())
        self.scheduler.start()
        self.logger.info("Maintainer Agent started with weekly schedule")

    def stop(self):
        """Stop maintenance activities"""
        self.scheduler.shutdown()
        self.logger.info("Maintainer Agent stopped")

    def _analyze_dependencies(self) -> Result[Dict[str, Any], Exception]:
        """Map function dependencies across codebase"""
        try:
            dependency_graph = self.dependency_analyzer.analyze_dependencies(self._get_python_files())
            return Success(dependency_graph)
        except Exception as e:
            return Failure(RuntimeError(f"Dependency analysis failed: {str(e)}"))

    def _verify_test_coverage(self):
        """Ensure new features have tests"""
        # coverage = pytest_cov.load() # This import is not used
        # The pytest_cov module is not installed, so this function will not work
        pass

    def _trigger_ci_pipeline(self):
        """Trigger CI/CD pipeline through GitHub API"""
        repo = self.github_client.get_repo(self.github_repo)
        # Tag the current commit with the active model version
        tag_name = f"model-version-{self.active_model_version}"
        try:
            # Get the latest commit SHA
            commits = repo.get_commits()
            latest_commit_sha = commits[0].sha
            # Create the tag
            repo.create_git_ref(ref=f"refs/tags/{tag_name}", sha=latest_commit_sha)
            self.logger.info(f"Created tag {tag_name} for model version {self.active_model_version}")
        except Exception as e:
            self.logger.error(f"Failed to create tag: {str(e)}")

        repo.create_repository_dispatch("main", {"event_type": "mlops-pipeline"})

    def _handle_rollback(self, version: str):
        """Rollback model version using git tags"""
        try:
            if not isinstance(version, str):
                raise ValueError("Version must be a string")
            repo = self.github_client.get_repo(self.github_repo)
            tag_name = f"model-version-{version}"
            try:
                # Get the tag
                tag = repo.get_git_ref(f"tags/{tag_name}")
                # Get the commit associated with the tag
                commit_sha = tag.object.sha
                commit = repo.get_commit(sha=commit_sha)

                # Reset the branch to the commit
                # This assumes you are rolling back the 'main' branch
                main_branch = repo.get_branch("main")
                main_branch.edit_commit(sha=commit.sha)

                self.logger.info(f"Rollback to model version {version} (commit {commit.sha}) successful")

            except Exception as e:
                self.logger.error(f"Failed to rollback to model version {version}: {str(e)}")
        except ValueError as e:
            self.logger.error(f"Invalid input: {str(e)}")

    def _get_previous_model_version(self):
        """Get the previous model version"""
        versions = list(self.model_performance.keys())
        versions.sort()  # Assuming versions are sortable
        if len(versions) > 1:
            return versions[-2]  # Return the second-to-last version
        else:
            return None

    def _run_accuracy_test(self) -> float:
        """Run accuracy test and return the accuracy"""
        # Implement your accuracy test logic here
        # Example:
        # accuracy = calculate_accuracy(self.model, self.test_data)
        return 0.95  # Replace with actual accuracy

    def _measure_inference_speed(self) -> float:
        """Measure inference speed and return the latency"""
        # Implement your latency measurement logic here
        # Example:
        # latency = measure_latency(self.model, self.test_data)
        return 1.5  # Replace with actual latency

    def _stress_test(self) -> int:
        """Run stress test and return the throughput"""
        # Implement your stress test logic here
        # Example:
        # throughput = run_stress_test(self.model)
        return 1000  # Replace with actual throughput

    def _get_all_functions(self) -> List[str]:
        """Get all function names in the codebase"""
        functions = []
        for name, obj in inspect.getmembers(self):
            if inspect.ismethod(obj):
                functions.append(name)
        return functions

    def _find_function_calls(self, function_name: str) -> set[str]:
        """Find all function calls within a function"""
        calls = set()
        source = inspect.getsource(getattr(self, function_name))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
        return calls

    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive data using Fernet"""
        return self.cipher_suite.encrypt(data.encode())

    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data using Fernet"""
        return self.cipher_suite.decrypt(encrypted_data).decode()

    def _parse_documentation(self, api_name: str) -> DocumentationAnalysisResult:
        """Enhanced doc analysis with AI-assisted parsing and state tracking"""
        doc_content = self._fetch_latest_documentation(api_name)
        return self.doc_analyzer.analyze_documentation(api_name, doc_content, self.doc_states, self._submit_github_issue)

    def _update_dependencies(self) -> bool:
        """Update project dependencies"""
        try:
            subprocess.check_call(["pip", "install", "--upgrade", "-r", "requirements.txt"])
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to update dependencies: {str(e)}")
            return False

    def _monitor_resource_usage(self) -> tuple[float, float]:
        """Monitor system resource usage"""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            return cpu_usage, memory_usage
        except Exception as e:
            self.logger.error(f"Failed to monitor resources: {str(e)}")
            return 0.0, 0.0

    def _get_last_analyzed_commit(self) -> Optional[str]:
        """Retrieves the last analyzed commit SHA from the state."""
        return self.state.last_analyzed_commit

    def _set_last_analyzed_commit(self, commit_sha: str):
        """Sets the last analyzed commit SHA in the state."""
        self.state.last_analyzed_commit = commit_sha

    def _run_in_sandbox(self, code: str, inputs: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any], Exception]:
        """Run code in a sandboxed environment.

        Args:
            code: Python code to execute
            inputs: Optional dictionary of input values

        Returns:
            Result containing execution metrics and output

        Raises:
            SecurityCheckError: If sandbox limits are exceeded
        """
        return self.code_sandbox.run_in_sandbox(code, inputs)

    def _perform_dynamic_analysis(self, code: str) -> Result[Dict[str, Any], Exception]:
        """Perform dynamic analysis on code.

        Args:
            code: Python code to analyze

        Returns:
            Result containing analysis metrics

        Raises:
            SecurityCheckError: If analysis fails
        """
        return self.dynamic_analyzer.perform_dynamic_analysis(code)

    def _analyze_test_coverage(self) -> Result[Dict[str, Any], Exception]:
        """Analyze test coverage and generate report.

        Returns:
            Result containing coverage metrics

        Raises:
            RuntimeError: If analysis fails
        """
        return self.test_coverage_analyzer.analyze_test_coverage()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "session"):
            await self.session.close()
