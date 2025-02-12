"""
Documentation Analysis Module

Handles parsing and analysis of API documentation.
"""

import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import yaml
from bleach import clean
import json
import requests
from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class DocumentationState:
    """Tracks the state of API documentation"""

    content_hash: str
    last_updated: datetime
    deprecated_functions: List[str]
    new_features: List[str]
    security_updates: List[str]
    breaking_changes: List[str]


class DocumentationAnalysisResult(BaseModel):
    """Result of documentation analysis"""

    deprecated: List[str]
    new_features: List[str]
    security: List[str]
    performance: List[str]
    breaking_changes: List[str]


class DocumentationAnalyzer:
    """Handles documentation analysis and tracking"""

    def __init__(self, openai_system, test_mode: bool = False):
        """Initialize the documentation analyzer.

        Args:
            openai_system: OpenAI system for RAG-based analysis
            test_mode: Whether to use mock responses
        """
        self.openai_system = openai_system
        self.test_mode = test_mode
        self.doc_states: Dict[str, DocumentationState] = {}
        self._load_states()

        # Load API URLs from config
        self.api_urls = Config.get_api_urls()

    def _load_states(self) -> None:
        """Load previous documentation states from disk"""
        try:
            with open(Config.DOC_STATES_PATH, "r") as f:
                states = yaml.safe_load(f)
                for api, state in states.items():
                    self.doc_states[api] = DocumentationState(**state)
        except FileNotFoundError:
            logger.info("No previous documentation states found")

    def _save_states(self) -> None:
        """Save current documentation states to disk"""
        try:
            states = {api: asdict(state) for api, state in self.doc_states.items()}
            with open(Config.DOC_STATES_PATH, "w") as f:
                yaml.dump(states, f)
        except Exception as e:
            logger.error(f"Failed to save documentation states: {str(e)}")
            raise

    def fetch_documentation(self, api_name: str) -> str:
        """Fetch latest API documentation.

        Args:
            api_name: Name of the API to fetch documentation for

        Returns:
            Documentation content as string

        Raises:
            ValueError: If API name is not recognized
            requests.RequestException: If documentation fetch fails
        """
        if self.test_mode:
            return "Test documentation content"

        if api_name not in self.api_urls:
            raise ValueError(f"Unknown API: {api_name}")

        try:
            headers = {"User-Agent": Config.USER_AGENT}
            response = requests.get(self.api_urls[api_name], headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {api_name} documentation: {str(e)}")
            raise

    def analyze_documentation(self, api_name: str) -> DocumentationAnalysisResult:
        """Analyze API documentation for changes.

        Args:
            api_name: Name of the API to analyze

        Returns:
            Analysis results containing changes

        Raises:
            ValueError: If API name is not recognized
        """
        try:
            # Fetch and sanitize content
            content = self.fetch_documentation(api_name)
            sanitized_content = clean(content)

            # Calculate content hash
            content_hash = hashlib.sha256(sanitized_content.encode()).hexdigest()

            # Check if content has changed
            prev_state = self.doc_states.get(api_name)
            if prev_state and prev_state.content_hash == content_hash:
                logger.info(f"No changes detected in {api_name} documentation")
                return DocumentationAnalysisResult(
                    deprecated=[], new_features=[], security=[], performance=[], breaking_changes=[]
                )

            # Analyze content with RAG
            analysis_prompt = f"""
            Analyze this {api_name} API documentation and identify:
            1. Deprecated functions/methods with removal timelines
            2. New features with version requirements
            3. Security recommendations
            4. Performance improvements
            5. Breaking changes
            
            Return JSON format with keys: deprecated, new_features, security, performance, breaking_changes
            """

            if self.test_mode:
                return DocumentationAnalysisResult(
                    deprecated=["mock_deprecated"],
                    new_features=["mock_feature"],
                    security=["mock_security"],
                    performance=["mock_performance"],
                    breaking_changes=["mock_breaking"],
                )

            response = self.openai_system.ask_with_rag(analysis_prompt, sanitized_content)
            result = DocumentationAnalysisResult(**json.loads(response))

            # Update state
            self.doc_states[api_name] = DocumentationState(
                content_hash=content_hash,
                last_updated=datetime.now(),
                deprecated_functions=result.deprecated,
                new_features=result.new_features,
                security_updates=result.security,
                breaking_changes=result.breaking_changes,
            )
            self._save_states()

            return result

        except Exception as e:
            logger.error(f"Documentation analysis failed for {api_name}: {str(e)}")
            raise

    def get_changes(self, api_name: str) -> Optional[Dict[str, List[str]]]:
        """Get changes between current and previous state.

        Args:
            api_name: Name of the API to check

        Returns:
            Dictionary of changes or None if no previous state
        """
        prev_state = self.doc_states.get(api_name)
        if not prev_state:
            return None

        try:
            current = self.analyze_documentation(api_name)

            return {
                "new_deprecations": list(set(current.deprecated) - set(prev_state.deprecated_functions)),
                "new_features": list(set(current.new_features) - set(prev_state.new_features)),
                "new_security": list(set(current.security) - set(prev_state.security_updates)),
                "new_breaking": list(set(current.breaking_changes) - set(prev_state.breaking_changes)),
            }

        except Exception as e:
            logger.error(f"Failed to get changes for {api_name}: {str(e)}")
            raise
