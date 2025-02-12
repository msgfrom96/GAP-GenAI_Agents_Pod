"""Static code analysis module."""

import ast
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from returns.result import Result, Success, Failure
import bleach

from ..config import Config
from .exceptions import CodeReviewError


@dataclass
class CodeIssue:
    """Container for code issues."""

    type: str
    severity: str
    file: str
    line: int
    message: str
    description: str
    recommendation: str


class CodeReviewer:
    """Handles static code analysis and review."""

    def __init__(self, github_manager, logger: Optional[logging.Logger] = None):
        """Initialize the code reviewer.

        Args:
            github_manager: GitHub manager for issue creation
            logger: Optional logger instance
        """
        self.github_manager = github_manager
        self.logger = logger or logging.getLogger(__name__)

    def review_code(self, code: str, filename: str) -> Result[List[CodeIssue], Exception]:
        """Perform static code analysis.

        Args:
            code: Code to analyze
            filename: Name of the file being analyzed

        Returns:
            Result containing list of code issues

        Raises:
            CodeReviewError: If analysis fails
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Collect all issues
            issues = []
            issues.extend(self._check_security_issues(tree, filename))
            issues.extend(self._check_code_style(tree, filename))
            issues.extend(self._check_complexity(tree, filename))
            issues.extend(self._check_documentation(tree, filename))
            issues.extend(self._check_error_handling(tree, filename))

            # Create GitHub issues for problems
            for issue in issues:
                self._create_issue(issue)

            return Success(issues)

        except SyntaxError as e:
            return Failure(CodeReviewError(f"Syntax error in {filename}: {str(e)}"))
        except Exception as e:
            self.logger.exception("Code review failed")
            return Failure(CodeReviewError(f"Code review failed: {str(e)}"))

    def _check_security_issues(self, tree: ast.AST, filename: str) -> List[CodeIssue]:
        """Check for security issues.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of security-related issues
        """
        issues = []

        for node in ast.walk(tree):
            # Check for subprocess usage
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ["call", "Popen", "run"]
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                ):

                    shell_arg = next((kw for kw in node.keywords if kw.arg == "shell"), None)
                    if shell_arg and isinstance(shell_arg.value, ast.Constant) and shell_arg.value.value:
                        issues.append(
                            CodeIssue(
                                type="security",
                                severity="high",
                                file=filename,
                                line=node.lineno,
                                message="Unsafe subprocess usage with shell=True",
                                description="Using shell=True with subprocess is dangerous as it can lead to command injection vulnerabilities.",
                                recommendation="Use shell=False and pass command arguments as a list.",
                            )
                        )

            # Check for hardcoded secrets
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id.lower()
                        if any(secret in name for secret in ["password", "secret", "key", "token"]):
                            if isinstance(node.value, ast.Constant):
                                issues.append(
                                    CodeIssue(
                                        type="security",
                                        severity="critical",
                                        file=filename,
                                        line=node.lineno,
                                        message="Hardcoded secret detected",
                                        description=f"Found hardcoded secret in variable {target.id}.",
                                        recommendation="Use environment variables or a secure secrets manager.",
                                    )
                                )

        return issues

    def _check_code_style(self, tree: ast.AST, filename: str) -> List[CodeIssue]:
        """Check code style.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of style-related issues
        """
        issues = []

        for node in ast.walk(tree):
            # Check function length
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > Config.MAX_FUNCTION_LENGTH:
                    issues.append(
                        CodeIssue(
                            type="style",
                            severity="medium",
                            file=filename,
                            line=node.lineno,
                            message="Function too long",
                            description=f"Function {node.name} has {len(node.body)} lines.",
                            recommendation="Break down into smaller functions.",
                        )
                    )

                # Check argument count
                if len(node.args.args) > Config.MAX_ARGUMENTS:
                    issues.append(
                        CodeIssue(
                            type="style",
                            severity="medium",
                            file=filename,
                            line=node.lineno,
                            message="Too many arguments",
                            description=f"Function {node.name} has {len(node.args.args)} arguments.",
                            recommendation="Use a data class or reduce argument count.",
                        )
                    )

            # Check line length
            if hasattr(node, "lineno"):
                source = ast.get_source_segment(code, node)
                if source and len(source) > Config.MAX_LINE_LENGTH:
                    issues.append(
                        CodeIssue(
                            type="style",
                            severity="low",
                            file=filename,
                            line=node.lineno,
                            message="Line too long",
                            description=f"Line exceeds {Config.MAX_LINE_LENGTH} characters.",
                            recommendation="Break line into multiple lines.",
                        )
                    )

        return issues

    def _check_complexity(self, tree: ast.AST, filename: str) -> List[CodeIssue]:
        """Check code complexity.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of complexity-related issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check cyclomatic complexity
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1

                if complexity > Config.MAX_COMPLEXITY:
                    issues.append(
                        CodeIssue(
                            type="complexity",
                            severity="high",
                            file=filename,
                            line=node.lineno,
                            message="Function too complex",
                            description=f"Function {node.name} has cyclomatic complexity of {complexity}.",
                            recommendation="Simplify logic or break into smaller functions.",
                        )
                    )

                # Check nesting depth
                max_depth = 0
                current_depth = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.With)):
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                    elif isinstance(child, ast.FunctionDef):
                        current_depth = 0

                if max_depth > Config.MAX_NESTING_DEPTH:
                    issues.append(
                        CodeIssue(
                            type="complexity",
                            severity="medium",
                            file=filename,
                            line=node.lineno,
                            message="Excessive nesting",
                            description=f"Function {node.name} has nesting depth of {max_depth}.",
                            recommendation="Reduce nesting by extracting logic into helper functions.",
                        )
                    )

        return issues

    def _check_documentation(self, tree: ast.AST, filename: str) -> List[CodeIssue]:
        """Check documentation.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of documentation-related issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Check for missing docstring
                if not ast.get_docstring(node):
                    issues.append(
                        CodeIssue(
                            type="documentation",
                            severity="medium",
                            file=filename,
                            line=node.lineno,
                            message="Missing docstring",
                            description=f"{node.__class__.__name__} {node.name} is missing a docstring.",
                            recommendation="Add a descriptive docstring.",
                        )
                    )
                else:
                    docstring = ast.get_docstring(node)
                    # Check docstring format
                    if isinstance(node, ast.FunctionDef):
                        if not re.search(r"Args:|Returns:|Raises:", docstring):
                            issues.append(
                                CodeIssue(
                                    type="documentation",
                                    severity="low",
                                    file=filename,
                                    line=node.lineno,
                                    message="Incomplete docstring",
                                    description=f"Function {node.name} docstring is missing sections.",
                                    recommendation="Add Args:, Returns:, and Raises: sections.",
                                )
                            )

        return issues

    def _check_error_handling(self, tree: ast.AST, filename: str) -> List[CodeIssue]:
        """Check error handling.

        Args:
            tree: AST of code to check
            filename: Name of file being checked

        Returns:
            List of error handling-related issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for missing error handling
                has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
                if not has_try and any(isinstance(n, ast.Call) for n in ast.walk(node)):
                    issues.append(
                        CodeIssue(
                            type="error_handling",
                            severity="medium",
                            file=filename,
                            line=node.lineno,
                            message="Missing error handling",
                            description=f"Function {node.name} makes calls but has no try-except blocks.",
                            recommendation="Add appropriate error handling.",
                        )
                    )

                # Check for bare except clauses
                for try_node in (n for n in ast.walk(node) if isinstance(n, ast.Try)):
                    if any(
                        isinstance(handler.type, ast.Name) and handler.type.id == "Exception" for handler in try_node.handlers
                    ):
                        issues.append(
                            CodeIssue(
                                type="error_handling",
                                severity="low",
                                file=filename,
                                line=try_node.lineno,
                                message="Bare except clause",
                                description="Using bare except Exception is too broad.",
                                recommendation="Catch specific exceptions.",
                            )
                        )

                    # Check for pass in except
                    for handler in try_node.handlers:
                        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                            issues.append(
                                CodeIssue(
                                    type="error_handling",
                                    severity="medium",
                                    file=filename,
                                    line=handler.lineno,
                                    message="Empty except block",
                                    description="Exception is caught but not handled.",
                                    recommendation="Handle the exception or document why it is ignored.",
                                )
                            )

        return issues

    def _create_issue(self, issue: CodeIssue) -> None:
        """Create a GitHub issue for a code problem.

        Args:
            issue: Code issue to report
        """
        title = f"Code Review: {issue.message}"
        body = f"""
        Code issue found in {issue.file} (line {issue.line})
        
        Severity: {issue.severity.upper()}
        Type: {issue.type}
        
        Description:
        {issue.description}
        
        Recommendation:
        {issue.recommendation}
        """

        # Sanitize input
        body = bleach.clean(body)

        labels = ["code-review", f"severity:{issue.severity}", f"type:{issue.type}"]

        self.github_manager.create_issue(title, body, labels)

    def review_diff(self, diff_data: str) -> Result[List[CodeIssue], Exception]:
        """
        Analyzes a diff (as a string) and returns a list of CodeIssues.
        """
        try:
            issues = []
            # Parse the diff data (you might need a library for this, or use regex)
            # For each changed file and line in the diff:
            #   - Extract the relevant code snippet.
            #   - Call self.analyze_code on the snippet.
            #   - Add any found issues to the 'issues' list.

            # This is a simplified example, assuming a simple diff format:
            for line in diff_data.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    # Extract the added code (simplified)
                    code_snippet = line[1:]
                    # You'll need to determine the filename and line number from the diff
                    # This is a placeholder.  A real diff parser is needed.
                    filename = "unknown_file.py"
                    line_number = 0  # Placeholder

                    analysis_result = self.review_code(code_snippet, filename)
                    if analysis_result.is_success():
                        issues.extend(analysis_result.unwrap())

            return Success(issues)
        except Exception as e:
            self.logger.exception(f"Error reviewing diff: {e}")
            return Failure(CodeReviewError(f"Error reviewing diff: {e}"))
