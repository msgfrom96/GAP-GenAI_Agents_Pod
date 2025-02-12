"""Dependency analysis module."""

import os
import json
import logging
import pkg_resources
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from returns.result import Result, Success, Failure
from packaging import version
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config
from .exceptions import DependencyError


@dataclass
class DependencyInfo:
    """Container for dependency information."""

    name: str
    current_version: str
    latest_version: Optional[str] = None
    is_outdated: bool = False
    vulnerabilities: List[Dict[str, Any]] = None


class DependencyAnalyzer:
    """Handles dependency analysis and vulnerability scanning."""

    def __init__(self, github_manager, logger: Optional[logging.Logger] = None):
        """Initialize the dependency analyzer.

        Args:
            github_manager: GitHub manager for issue creation
            logger: Optional logger instance
        """
        self.github_manager = github_manager
        self.logger = logger or logging.getLogger(__name__)
        self._load_vulnerability_db()

    def analyze_dependencies(self) -> Result[Dict[str, Any], Exception]:
        """Analyze project dependencies.

        Returns:
            Result containing dependency analysis results

        Raises:
            DependencyError: If analysis fails
        """
        try:
            results = {"outdated": [], "vulnerabilities": [], "dependencies": []}

            # Check each installed package
            for pkg in pkg_resources.working_set:
                pkg_info = self._analyze_package(pkg)

                if pkg_info.is_outdated:
                    results["outdated"].append(
                        {
                            "package": pkg_info.name,
                            "current_version": pkg_info.current_version,
                            "latest_version": pkg_info.latest_version,
                        }
                    )

                if pkg_info.vulnerabilities:
                    results["vulnerabilities"].extend(
                        {"package": pkg_info.name, "version": pkg_info.current_version, "vulnerability": vuln}
                        for vuln in pkg_info.vulnerabilities
                    )

                results["dependencies"].append(
                    {
                        "name": pkg_info.name,
                        "current_version": pkg_info.current_version,
                        "latest_version": pkg_info.latest_version,
                        "is_outdated": pkg_info.is_outdated,
                    }
                )

            # Save results to cache
            Config.save_state(Config.DEPENDENCY_CACHE, results)

            # Create issues for problems
            self._create_dependency_issues(results)

            return Success(results)

        except Exception as e:
            self.logger.exception("Dependency analysis failed")
            return Failure(DependencyError(f"Dependency analysis failed: {str(e)}"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _analyze_package(self, pkg: pkg_resources.Distribution) -> DependencyInfo:
        """Analyze a single package.

        Args:
            pkg: Package to analyze

        Returns:
            Package analysis information

        Raises:
            DependencyError: If package analysis fails
        """
        try:
            # Get package info
            info = DependencyInfo(name=pkg.key, current_version=pkg.version, vulnerabilities=[])

            # Check PyPI for latest version
            response = requests.get(f"https://pypi.org/pypi/{pkg.key}/json", timeout=Config.REQUEST_TIMEOUT)
            if response.status_code == 200:
                pypi_data = response.json()
                latest_version = pypi_data["info"]["version"]
                info.latest_version = latest_version

                # Check if outdated
                if version.parse(latest_version) > version.parse(pkg.version):
                    info.is_outdated = True

            # Check for known vulnerabilities
            if pkg.key in self.vulnerability_db:
                for vuln in self.vulnerability_db[pkg.key]:
                    if version.parse(pkg.version) in version.parse(vuln["affected_versions"]):
                        info.vulnerabilities.append(vuln)

            return info

        except Exception as e:
            self.logger.error(f"Failed to analyze package {pkg.key}: {str(e)}")
            raise DependencyError(f"Failed to analyze package {pkg.key}: {str(e)}")

    def _load_vulnerability_db(self) -> None:
        """Load vulnerability database."""
        try:
            if os.path.exists(Config.VULNERABILITY_DB_PATH):
                with open(Config.VULNERABILITY_DB_PATH) as f:
                    self.vulnerability_db = json.load(f)
            else:
                self.vulnerability_db = {}

        except Exception as e:
            self.logger.error(f"Failed to load vulnerability database: {str(e)}")
            self.vulnerability_db = {}

    def _create_dependency_issues(self, results: Dict[str, Any]) -> None:
        """Create GitHub issues for dependency problems.

        Args:
            results: Dependency analysis results
        """
        # Report outdated packages
        if results["outdated"]:
            title = "Outdated Dependencies Detected"
            body = "The following packages are outdated:\n\n"
            for pkg in results["outdated"]:
                body += f"- {pkg['package']}: {pkg['current_version']} -> {pkg['latest_version']}\n"

            self.github_manager.create_issue(title=title, body=body, labels=["dependencies", "outdated"])

        # Report vulnerabilities
        if results["vulnerabilities"]:
            title = "Security Vulnerabilities in Dependencies"
            body = "The following packages have known vulnerabilities:\n\n"
            for vuln in results["vulnerabilities"]:
                body += f"### {vuln['package']} {vuln['version']}\n"
                body += f"**Vulnerability**: {vuln['vulnerability']['description']}\n"
                if "severity" in vuln["vulnerability"]:
                    body += f"**Severity**: {vuln['vulnerability']['severity']}\n"
                if "fix_versions" in vuln["vulnerability"]:
                    body += f"**Fix Versions**: {', '.join(vuln['vulnerability']['fix_versions'])}\n"
                body += "\n"

            self.github_manager.create_issue(title=title, body=body, labels=["dependencies", "security", "vulnerability"])

    def update_dependencies(self) -> Result[bool, Exception]:
        """Update project dependencies.

        Returns:
            Result indicating success

        Raises:
            DependencyError: If update fails
        """
        try:
            # Run pip install --upgrade
            import subprocess

            subprocess.check_call(["pip", "install", "--upgrade", "-r", "requirements.txt"])

            # Re-analyze dependencies
            analysis_result = self.analyze_dependencies()
            if analysis_result.is_failure():
                return analysis_result

            return Success(True)

        except Exception as e:
            self.logger.exception("Dependency update failed")
            return Failure(DependencyError(f"Dependency update failed: {str(e)}"))

    def check_compatibility(self, package: str, version: str) -> Result[bool, Exception]:
        """Check if a package version is compatible.

        Args:
            package: Package name
            version: Version to check

        Returns:
            Result indicating compatibility

        Raises:
            DependencyError: If check fails
        """
        try:
            # Get current dependencies
            with open("requirements.txt") as f:
                requirements = [line.strip() for line in f if line.strip()]

            # Check each requirement
            for req in requirements:
                try:
                    name, spec = req.split("==")
                    if name == package and version != spec:
                        return Success(False)
                except ValueError:
                    continue

            return Success(True)

        except Exception as e:
            self.logger.exception("Compatibility check failed")
            return Failure(DependencyError(f"Compatibility check failed: {str(e)}"))

    def get_dependency_tree(self) -> Result[Dict[str, List[str]], Exception]:
        """Get dependency tree.

        Returns:
            Result containing dependency tree

        Raises:
            DependencyError: If tree generation fails
        """
        try:
            tree = {}

            for pkg in pkg_resources.working_set:
                tree[pkg.key] = [str(dep) for dep in pkg.requires()]

            return Success(tree)

        except Exception as e:
            self.logger.exception("Failed to generate dependency tree")
            return Failure(DependencyError(f"Failed to generate dependency tree: {str(e)}"))

    def check_licenses(self) -> Result[Dict[str, str], Exception]:
        """Check package licenses.

        Returns:
            Result containing package licenses

        Raises:
            DependencyError: If license check fails
        """
        try:
            licenses = {}

            for pkg in pkg_resources.working_set:
                try:
                    meta = pkg.get_metadata("METADATA")
                    for line in meta.split("\n"):
                        if line.startswith("License:"):
                            licenses[pkg.key] = line.split(":", 1)[1].strip()
                            break
                except Exception:
                    licenses[pkg.key] = "Unknown"

            return Success(licenses)

        except Exception as e:
            self.logger.exception("License check failed")
            return Failure(DependencyError(f"License check failed: {str(e)}"))
