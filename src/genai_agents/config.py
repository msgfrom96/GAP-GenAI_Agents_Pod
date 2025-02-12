"""
Configuration Module

Handles all configuration settings for the application.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml
import json
import logging


@dataclass
class AIServiceConfig:
    """Configuration for AI services"""

    api_key: str
    model_name: str
    embedding_model: str


class Config:
    """Central configuration management"""

    # File paths
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    DOC_STATES_PATH = os.path.join(CONFIG_DIR, "doc_states.yaml")
    PRICING_DATA_PATH = os.path.join(CONFIG_DIR, "pricing_data.json")
    SECURITY_RULES_PATH = os.path.join(CONFIG_DIR, "security_rules.yaml")
    STATE_DIR = os.path.join(CONFIG_DIR, "state")
    CACHE_DIR = os.path.join(CONFIG_DIR, "cache")
    SANDBOX_DIR = os.path.join(CONFIG_DIR, "sandbox")
    TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
    DEPLOY_TESTS = os.getenv("DEPLOY_TESTS", "0") == "1"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # State files
    LAST_COMMIT_FILE = os.path.join(STATE_DIR, "last_commit.txt")
    DEPENDENCY_CACHE = os.path.join(CACHE_DIR, "dependencies.json")
    COVERAGE_HISTORY = os.path.join(STATE_DIR, "coverage_history.json")
    ANALYSIS_RESULTS = os.path.join(STATE_DIR, "analysis_results.json")

    # Sandbox configuration
    SANDBOX_PYTHON = os.getenv("SANDBOX_PYTHON", "python3")
    SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))  # seconds
    SANDBOX_MEMORY_LIMIT = int(os.getenv("SANDBOX_MEMORY_LIMIT", "512"))  # MB

    # Dynamic analysis settings
    DYNAMIC_ANALYSIS_TIMEOUT = int(os.getenv("DYNAMIC_ANALYSIS_TIMEOUT", "60"))
    DYNAMIC_ANALYSIS_MAX_ITERATIONS = int(os.getenv("DYNAMIC_ANALYSIS_MAX_ITERATIONS", "1000"))

    # Dependency analysis
    DEPENDENCY_CHECK_INTERVAL = int(os.getenv("DEPENDENCY_CHECK_INTERVAL", "86400"))  # 24 hours
    VULNERABILITY_DB_PATH = os.path.join(CONFIG_DIR, "vulnerability_db.json")

    # Test coverage thresholds
    COVERAGE_THRESHOLDS = {
        "min_total": float(os.getenv("MIN_COVERAGE", "80")),
        "min_unit": float(os.getenv("MIN_UNIT_COVERAGE", "85")),
        "min_integration": float(os.getenv("MIN_INTEGRATION_COVERAGE", "75")),
        "critical_paths": float(os.getenv("CRITICAL_PATH_COVERAGE", "90")),
    }

    # Resource limits
    RESOURCE_LIMITS = {
        "max_cpu_percent": float(os.getenv("MAX_CPU_PERCENT", "80")),
        "max_memory_percent": float(os.getenv("MAX_MEMORY_PERCENT", "80")),
        "max_disk_percent": float(os.getenv("MAX_DISK_PERCENT", "90")),
    }

    # API URLs
    API_URLS = {
        "OpenAI": "https://raw.githubusercontent.com/openai/openai-python/main/README.md",
        "Google": "https://ai.google.dev/api/python/google/generativeai",
    }

    # Pricing URLs
    PRICING_URLS = {"OpenAI": "https://platform.openai.com/docs/pricing", "Google": "https://ai.google.dev/pricing"}

    # GitHub settings
    GITHUB_REPO = os.getenv("GITHUB_REPO", "your-org/your-repo")
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

    MAX_ISSUE_TITLE_LENGTH = 256
    MAX_ISSUE_BODY_LENGTH = 65536

    # HTTP settings
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    REQUEST_TIMEOUT = 10

    # Model performance thresholds
    MODEL_THRESHOLDS = {"accuracy": 0.85, "latency": 2.0, "min_throughput": 100}

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.CONFIG_DIR, cls.STATE_DIR, cls.CACHE_DIR, cls.SANDBOX_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def load_state(cls, state_file: str) -> Dict[str, Any]:
        """Load state from a JSON file.

        Args:
            state_file: Path to the state file

        Returns:
            Dictionary containing state data
        """
        try:
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Failed to load state from {state_file}: {e}")
            return {}

    @classmethod
    def save_state(cls, state_file: str, data: Dict[str, Any]) -> None:
        """Save state to a JSON file.

        Args:
            state_file: Path to the state file
            data: Dictionary of state data to save
        """
        try:
            cls.ensure_directories()
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save state to {state_file}: {e}")

    @classmethod
    def get_last_analyzed_commit(cls) -> Optional[str]:
        """Get the SHA of the last analyzed commit."""
        try:
            if os.path.exists(cls.LAST_COMMIT_FILE):
                with open(cls.LAST_COMMIT_FILE, "r") as f:
                    return f.read().strip()
            return None
        except Exception as e:
            logging.error(f"Failed to read last commit: {e}")
            return None

    @classmethod
    def set_last_analyzed_commit(cls, commit_sha: str) -> None:
        """Save the SHA of the last analyzed commit."""
        try:
            cls.ensure_directories()
            with open(cls.LAST_COMMIT_FILE, "w") as f:
                f.write(commit_sha)
        except Exception as e:
            logging.error(f"Failed to save last commit: {e}")

    @classmethod
    def load_openai_config(cls) -> AIServiceConfig:
        """Load OpenAI configuration.

        Returns:
            OpenAI service configuration

        Raises:
            ValueError: If required environment variables are missing
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return AIServiceConfig(
            api_key=api_key,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        )

    @classmethod
    def load_google_config(cls) -> AIServiceConfig:
        """Load Google AI configuration.

        Returns:
            Google AI service configuration

        Raises:
            ValueError: If required environment variables are missing
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        return AIServiceConfig(
            api_key=api_key,
            model_name=os.getenv("GOOGLE_MODEL", "gemini-pro"),
            embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004"),
        )

    @classmethod
    def get_github_token(cls) -> Optional[str]:
        """Get GitHub access token.

        Returns:
            GitHub token or None if not set
        """
        return os.getenv("GITHUB_ACCESS_TOKEN")

    @classmethod
    def get_api_urls(cls) -> Dict[str, str]:
        """Get API documentation URLs.

        Returns:
            Dictionary mapping API names to documentation URLs
        """
        return cls.API_URLS.copy()

    @classmethod
    def get_pricing_urls(cls) -> Dict[str, str]:
        """Get API pricing URLs.

        Returns:
            Dictionary mapping API names to pricing URLs
        """
        return cls.PRICING_URLS.copy()

    @classmethod
    def get_model_thresholds(cls) -> Dict[str, float]:
        """Get model performance thresholds.

        Returns:
            Dictionary of threshold values
        """
        return cls.MODEL_THRESHOLDS.copy()

    @classmethod
    def get_security_rules(cls) -> Dict[str, Any]:
        """Load security rules from configuration.

        Returns:
            Dictionary of security rules

        Raises:
            FileNotFoundError: If rules file doesn't exist
            yaml.YAMLError: If rules file is invalid
        """
        try:
            with open(cls.SECURITY_RULES_PATH, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "subprocess": {"severity": "high", "patterns": ["subprocess.call", "subprocess.Popen"]},
                "deserialization": {"severity": "high", "patterns": ["pickle.load", "yaml.load"]},
                "sql_injection": {"severity": "critical", "patterns": ["execute(", "executemany("]},
                "authentication": {"severity": "high", "required_decorators": ["login_required"]},
                "csrf": {"severity": "high", "required_decorators": ["csrf_protect"]},
                "xss": {"severity": "high", "patterns": ["render(", "render_template("]},
            }

    @classmethod
    def setup_logging(cls) -> None:
        """Configure logging settings."""
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Set up file handler
        log_file = os.path.join(logs_dir, "app.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    @classmethod
    def create_config_files(cls) -> None:
        """Create default configuration files if they don't exist."""
        # Create config directory
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)

        # Create security rules file
        if not os.path.exists(cls.SECURITY_RULES_PATH):
            rules = cls.get_security_rules()
            with open(cls.SECURITY_RULES_PATH, "w") as f:
                yaml.dump(rules, f)

        # Create empty doc states file
        if not os.path.exists(cls.DOC_STATES_PATH):
            with open(cls.DOC_STATES_PATH, "w") as f:
                yaml.dump({}, f)

        # Create empty pricing data file
        if not os.path.exists(cls.PRICING_DATA_PATH):
            with open(cls.PRICING_DATA_PATH, "w") as f:
                f.write("{}")

    @classmethod
    def validate_environment(cls) -> None:
        """Validate required environment variables.

        Raises:
            ValueError: If required variables are missing
        """
        required_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GITHUB_ACCESS_TOKEN"]

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    @classmethod
    def initialize(cls) -> None:
        """Initialize configuration.

        This:
        1. Creates config directory and files
        2. Sets up logging
        3. Validates environment

        Raises:
            ValueError: If initialization fails
        """
        try:
            cls.create_config_files()
            cls.setup_logging()
            cls.validate_environment()
        except Exception as e:
            raise ValueError(f"Failed to initialize configuration: {str(e)}")
