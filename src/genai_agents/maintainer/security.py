"""
Security Compliance Module

Handles security checks and vulnerability scanning.
"""

import logging
import re
from typing import List, Dict, Any, Optional
import ast
from ..config import Config

logger = logging.getLogger(__name__)


class SecurityChecker:
    """Handles security compliance checks and vulnerability scanning"""

    def __init__(self, github_manager, test_mode: bool = False):
        """Initialize the security checker.

        Args:
            github_manager: GitHub manager for issue creation
            test_mode: Whether to use mock responses
        """
        self.github_manager = github_manager
        self.test_mode = test_mode

        # Load security rules from config
        self.rules = Config.get_security_rules()

    def check_code_security(self, code: str, filename: str) -> List[Dict[str, Any]]:
        """Check code for security issues.

        Args:
            code: Code to check
            filename: Name of the file being checked

        Returns:
            List of security issues found

        Raises:
            ValueError: If code is empty
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        issues = []

        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Check for various security issues
            issues.extend(self._check_subprocess_usage(tree, filename))
            issues.extend(self._check_deserialization(tree, filename))
            issues.extend(self._check_input_validation(tree, filename))
            issues.extend(self._check_error_handling(tree, filename))
            issues.extend(self._check_encryption(tree, filename))
            issues.extend(self._check_authentication(tree, filename))

            # Create GitHub issues for security problems
            for issue in issues:
                self._create_security_issue(issue)

            return issues

        except SyntaxError as e:
            logger.error(f"Syntax error in {filename}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Failed to check code security: {str(e)}")
            raise RuntimeError(f"Failed to check code security: {str(e)}")

    def _check_subprocess_usage(self, tree: ast.AST, filename: str) -> List[Dict[str, Any]]:
        """Check for unsafe subprocess usage.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of subprocess-related security issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ["call", "Popen", "run"]
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                ):

                    # Check for shell=True
                    shell_arg = next((kw for kw in node.keywords if kw.arg == "shell"), None)
                    if shell_arg and isinstance(shell_arg.value, ast.Constant) and shell_arg.value.value:
                        issues.append(
                            {
                                "type": "subprocess",
                                "severity": "high",
                                "file": filename,
                                "line": node.lineno,
                                "message": "Unsafe subprocess usage with shell=True",
                                "description": "Using shell=True with subprocess is dangerous as it can lead to command injection vulnerabilities.",
                            }
                        )

        return issues

    def _check_deserialization(self, tree: ast.AST, filename: str) -> List[Dict[str, Any]]:
        """Check for unsafe deserialization.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of deserialization-related security issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "load" and isinstance(node.func.value, ast.Name):

                    if node.func.value.id in ["pickle", "yaml"]:
                        issues.append(
                            {
                                "type": "deserialization",
                                "severity": "high",
                                "file": filename,
                                "line": node.lineno,
                                "message": f"Unsafe {node.func.value.id} deserialization",
                                "description": f"Using {node.func.value.id}.load is dangerous as it can lead to code execution vulnerabilities.",
                            }
                        )

        return issues

    def _check_input_validation(self, tree: ast.AST, filename: str) -> List[Dict[str, Any]]:
        """Check for missing input validation.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of input validation-related security issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has parameters but no type hints
                if node.args.args and not all(arg.annotation for arg in node.args.args):
                    issues.append(
                        {
                            "type": "input_validation",
                            "severity": "medium",
                            "file": filename,
                            "line": node.lineno,
                            "message": "Missing type hints",
                            "description": f"Function {node.name} has parameters without type hints.",
                        }
                    )

                # Check for input validation in function body
                if not any(isinstance(n, ast.If) for n in ast.walk(node)):
                    issues.append(
                        {
                            "type": "input_validation",
                            "severity": "medium",
                            "file": filename,
                            "line": node.lineno,
                            "message": "Missing input validation",
                            "description": f"Function {node.name} may not validate its inputs.",
                        }
                    )

        return issues

    def _check_error_handling(self, tree: ast.AST, filename: str) -> List[Dict[str, Any]]:
        """Check for missing error handling.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of error handling-related security issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for try-except blocks
                has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
                if not has_try:
                    issues.append(
                        {
                            "type": "error_handling",
                            "severity": "medium",
                            "file": filename,
                            "line": node.lineno,
                            "message": "Missing error handling",
                            "description": f"Function {node.name} does not have any try-except blocks.",
                        }
                    )

                # Check for bare except clauses
                for try_node in (n for n in ast.walk(node) if isinstance(n, ast.Try)):
                    if any(
                        isinstance(handler.type, ast.Name) and handler.type.id == "Exception" for handler in try_node.handlers
                    ):
                        issues.append(
                            {
                                "type": "error_handling",
                                "severity": "low",
                                "file": filename,
                                "line": try_node.lineno,
                                "message": "Bare except clause",
                                "description": "Using bare except Exception is too broad and may hide bugs.",
                            }
                        )

        return issues

    def _check_encryption(self, tree: ast.AST, filename: str) -> List[Dict[str, Any]]:
        """Check for proper encryption usage.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of encryption-related security issues
        """
        issues = []

        # Check for hardcoded secrets
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id.lower()
                        if any(secret in name for secret in ["password", "secret", "key", "token"]):
                            if isinstance(node.value, ast.Constant):
                                issues.append(
                                    {
                                        "type": "encryption",
                                        "severity": "critical",
                                        "file": filename,
                                        "line": node.lineno,
                                        "message": "Hardcoded secret",
                                        "description": f"Found hardcoded secret in variable {target.id}.",
                                    }
                                )

        return issues

    def _check_authentication(self, tree: ast.AST, filename: str) -> List[Dict[str, Any]]:
        """Check for proper authentication.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of authentication-related security issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for @login_required decorator
                if any("api" in decorator.id.lower() for decorator in node.decorator_list if isinstance(decorator, ast.Name)):
                    if not any(
                        "login_required" in decorator.id.lower()
                        for decorator in node.decorator_list
                        if isinstance(decorator, ast.Name)
                    ):
                        issues.append(
                            {
                                "type": "authentication",
                                "severity": "high",
                                "file": filename,
                                "line": node.lineno,
                                "message": "Missing authentication",
                                "description": f"API function {node.name} is missing @login_required decorator.",
                            }
                        )

        return issues

    def _create_security_issue(self, issue: Dict[str, Any]) -> None:
        """Create a GitHub issue for a security problem.

        Args:
            issue: Dictionary containing issue details
        """
        title = f"Security Warning: {issue['message']}"
        body = f"""
        Security issue found in {issue['file']} (line {issue['line']})
        
        Severity: {issue['severity'].upper()}
        Type: {issue['type']}
        
        Description:
        {issue['description']}
        
        Please review and fix this security issue as soon as possible.
        """

        labels = ["security", f'severity:{issue["severity"]}']

        if not self.test_mode:
            self.github_manager.create_issue(title, body, labels)

    def scan_dependencies(self) -> List[Dict[str, Any]]:
        """Scan dependencies for known vulnerabilities.

        Returns:
            List of vulnerability reports

        Raises:
            RuntimeError: If scan fails
        """
        if self.test_mode:
            return []

        try:
            # This would typically use a tool like safety or snyk
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Failed to scan dependencies: {str(e)}")
            raise RuntimeError(f"Failed to scan dependencies: {str(e)}")

    def check_file_permissions(self, filename: str) -> List[Dict[str, Any]]:
        """Check file permissions.

        Args:
            filename: File to check

        Returns:
            List of permission-related issues

        Raises:
            RuntimeError: If check fails
        """
        if self.test_mode:
            return []

        try:
            # This would check file permissions
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Failed to check file permissions: {str(e)}")
            raise RuntimeError(f"Failed to check file permissions: {str(e)}")

    def check_api_security(self, code: str, filename: str) -> List[Dict[str, Any]]:
        """Check API endpoint security.

        Args:
            code: Code to check
            filename: Name of file being checked

        Returns:
            List of API security issues

        Raises:
            ValueError: If code is empty
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for API endpoints
                    if any(
                        "route" in decorator.id.lower() for decorator in node.decorator_list if isinstance(decorator, ast.Name)
                    ):

                        # Check authentication
                        if not any(
                            "login_required" in decorator.id.lower()
                            for decorator in node.decorator_list
                            if isinstance(decorator, ast.Name)
                        ):
                            issues.append(
                                {
                                    "type": "api_security",
                                    "severity": "high",
                                    "file": filename,
                                    "line": node.lineno,
                                    "message": "Unauthenticated API endpoint",
                                    "description": f"API endpoint {node.name} is not protected by authentication.",
                                }
                            )

                        # Check CSRF protection
                        if not any(
                            "csrf_protect" in decorator.id.lower()
                            for decorator in node.decorator_list
                            if isinstance(decorator, ast.Name)
                        ):
                            issues.append(
                                {
                                    "type": "api_security",
                                    "severity": "high",
                                    "file": filename,
                                    "line": node.lineno,
                                    "message": "Missing CSRF protection",
                                    "description": f"API endpoint {node.name} is not protected against CSRF attacks.",
                                }
                            )

            return issues

        except Exception as e:
            logger.error(f"Failed to check API security: {str(e)}")
            raise RuntimeError(f"Failed to check API security: {str(e)}")

    def check_sql_injection(self, code: str, filename: str) -> List[Dict[str, Any]]:
        """Check for SQL injection vulnerabilities.

        Args:
            code: Code to check
            filename: Name of file being checked

        Returns:
            List of SQL injection issues

        Raises:
            ValueError: If code is empty
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in ["execute", "executemany"]
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in ["cursor", "connection"]
                    ):

                        # Check for string formatting or concatenation
                        if len(node.args) > 0 and isinstance(node.args[0], ast.BinOp):
                            issues.append(
                                {
                                    "type": "sql_injection",
                                    "severity": "critical",
                                    "file": filename,
                                    "line": node.lineno,
                                    "message": "Potential SQL injection",
                                    "description": "SQL query uses string concatenation instead of parameterized queries.",
                                }
                            )

            return issues

        except Exception as e:
            logger.error(f"Failed to check for SQL injection: {str(e)}")
            raise RuntimeError(f"Failed to check for SQL injection: {str(e)}")

    def check_xss(self, code: str, filename: str) -> List[Dict[str, Any]]:
        """Check for XSS vulnerabilities.

        Args:
            code: Code to check
            filename: Name of file being checked

        Returns:
            List of XSS issues

        Raises:
            ValueError: If code is empty
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "render"
                        and isinstance(node.func.value, ast.Name)
                    ):

                        # Check for unescaped variables in templates
                        for kw in node.keywords:
                            if isinstance(kw.value, ast.Name):
                                issues.append(
                                    {
                                        "type": "xss",
                                        "severity": "high",
                                        "file": filename,
                                        "line": node.lineno,
                                        "message": "Potential XSS vulnerability",
                                        "description": f"Template variable {kw.value.id} may not be properly escaped.",
                                    }
                                )

            return issues

        except Exception as e:
            logger.error(f"Failed to check for XSS: {str(e)}")
            raise RuntimeError(f"Failed to check for XSS: {str(e)}")
