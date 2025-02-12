"""State management module."""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from returns.result import Result, Success, Failure

from ..config import Config
from .exceptions import StateError


@dataclass
class State:
    """Container for state data."""

    last_analyzed_commit: Optional[str] = None
    last_run: Optional[str] = None
    duration: Optional[float] = None
    phases_completed: int = 0
    issues_found: int = 0
    model_performance: Dict[str, Dict[str, float]] = None
    doc_states: Dict[str, Dict[str, Any]] = None
    pricing_data: Dict[str, Dict[str, Any]] = None
    dependency_cache: Dict[str, Dict[str, Any]] = None
    coverage_history: Dict[str, Dict[str, float]] = None


class StateManager:
    """Handles persistent state management."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the state manager.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._ensure_directories()

    def load_state(self) -> Result[State, Exception]:
        """Load state from disk.

        Returns:
            Result containing state data

        Raises:
            StateError: If state loading fails
        """
        try:
            state = State()

            # Load last analyzed commit
            state.last_analyzed_commit = Config.get_last_analyzed_commit()

            # Load analysis results
            analysis_results = Config.load_state(Config.ANALYSIS_RESULTS)
            if analysis_results:
                state.last_run = analysis_results.get("last_run")
                state.duration = analysis_results.get("duration")
                state.phases_completed = analysis_results.get("phases_completed", 0)
                state.issues_found = analysis_results.get("issues_found", 0)

            # Load doc states
            try:
                with open(Config.DOC_STATES_PATH, "r") as f:
                    state.doc_states = yaml.safe_load(f) or {}
            except FileNotFoundError:
                state.doc_states = {}

            # Load pricing data
            try:
                with open(Config.PRICING_DATA_PATH, "r") as f:
                    state.pricing_data = json.load(f) or {}
            except FileNotFoundError:
                state.pricing_data = {}

            # Load dependency cache
            state.dependency_cache = Config.load_state(Config.DEPENDENCY_CACHE)

            # Load coverage history
            state.coverage_history = Config.load_state(Config.COVERAGE_HISTORY)

            return Success(state)

        except Exception as e:
            return Failure(StateError(f"Failed to load state: {str(e)}"))

    def save_state(self, state: State) -> Result[bool, Exception]:
        """Save state to disk.

        Args:
            state: State object to save

        Returns:
            Result indicating success

        Raises:
            StateError: If state saving fails
        """
        try:
            # Save last analyzed commit
            if state.last_analyzed_commit:
                Config.set_last_analyzed_commit(state.last_analyzed_commit)

            # Save analysis results
            Config.save_state(
                Config.ANALYSIS_RESULTS,
                {
                    "last_run": state.last_run,
                    "duration": state.duration,
                    "phases_completed": state.phases_completed,
                    "issues_found": state.issues_found,
                },
            )

            # Save doc states
            with open(Config.DOC_STATES_PATH, "w") as f:
                yaml.dump(state.doc_states, f)

            # Save pricing data
            with open(Config.PRICING_DATA_PATH, "w") as f:
                json.dump(state.pricing_data, f, indent=2)

            # Save dependency cache
            Config.save_state(Config.DEPENDENCY_CACHE, state.dependency_cache)

            # Save coverage history
            Config.save_state(Config.COVERAGE_HISTORY, state.coverage_history)

            return Success(True)

        except Exception as e:
            return Failure(StateError(f"Failed to save state: {str(e)}"))

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        Config.ensure_directories()

    def clear_state(self) -> Result[bool, Exception]:
        """Clear all state data.

        Returns:
            Result indicating success

        Raises:
            StateError: If state clearing fails
        """
        try:
            # Remove state files
            files_to_remove = [
                Config.LAST_COMMIT_FILE,
                Config.ANALYSIS_RESULTS,
                Config.DOC_STATES_PATH,
                Config.PRICING_DATA_PATH,
                Config.DEPENDENCY_CACHE,
                Config.COVERAGE_HISTORY,
            ]

            for file in files_to_remove:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass

            return Success(True)

        except Exception as e:
            return Failure(StateError(f"Failed to clear state: {str(e)}"))

    def get_last_analyzed_commit(self) -> Result[Optional[str], Exception]:
        """Get the SHA of the last analyzed commit.

        Returns:
            Result containing commit SHA or None

        Raises:
            StateError: If commit retrieval fails
        """
        try:
            commit = Config.get_last_analyzed_commit()
            return Success(commit)
        except Exception as e:
            return Failure(StateError(f"Failed to get last commit: {str(e)}"))

    def set_last_analyzed_commit(self, commit_sha: str) -> Result[bool, Exception]:
        """Save the SHA of the last analyzed commit.

        Args:
            commit_sha: Commit SHA to save

        Returns:
            Result indicating success

        Raises:
            StateError: If commit saving fails
        """
        try:
            Config.set_last_analyzed_commit(commit_sha)
            return Success(True)
        except Exception as e:
            return Failure(StateError(f"Failed to save last commit: {str(e)}"))

    def update_model_performance(self, version: str, metrics: Dict[str, float]) -> Result[bool, Exception]:
        """Update model performance metrics.

        Args:
            version: Model version
            metrics: Performance metrics

        Returns:
            Result indicating success

        Raises:
            StateError: If update fails
        """
        try:
            state_result = self.load_state()
            if state_result.is_failure():
                return state_result

            state = state_result.unwrap()
            if state.model_performance is None:
                state.model_performance = {}

            state.model_performance[version] = metrics

            return self.save_state(state)

        except Exception as e:
            return Failure(StateError(f"Failed to update model performance: {str(e)}"))

    def get_model_performance(self, version: Optional[str] = None) -> Result[Dict[str, Dict[str, float]], Exception]:
        """Get model performance metrics.

        Args:
            version: Optional specific version to get

        Returns:
            Result containing performance metrics

        Raises:
            StateError: If retrieval fails
        """
        try:
            state_result = self.load_state()
            if state_result.is_failure():
                return state_result

            state = state_result.unwrap()
            if state.model_performance is None:
                return Success({})

            if version:
                return Success({version: state.model_performance.get(version, {})})
            return Success(state.model_performance)

        except Exception as e:
            return Failure(StateError(f"Failed to get model performance: {str(e)}"))
