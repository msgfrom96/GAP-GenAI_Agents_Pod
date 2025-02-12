"""
GitHub Integration Module

Handles GitHub issue tracking and repository management.
"""

import logging
from typing import Optional, List, Dict, Any
from github import Github
from github.Repository import Repository
from github.Issue import Issue
from ..config import Config

logger = logging.getLogger(__name__)


class GitHubManager:
    """Handles GitHub integration and issue management"""

    def __init__(self, test_mode: bool = False):
        """Initialize the GitHub manager.

        Args:
            test_mode: Whether to use mock responses
        """
        self.test_mode = test_mode

        if not test_mode:
            github_token = Config.get_github_token()
            if not github_token:
                raise ValueError("GitHub token not found in configuration")
            self.client = Github(github_token)
            self.repo = self.client.get_repo(Config.GITHUB_REPO)
        else:
            self.client = None
            self.repo = None

    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None) -> Optional[Issue]:
        """Create a GitHub issue.

        Args:
            title: Issue title
            body: Issue description
            labels: Optional list of labels to apply

        Returns:
            Created issue or None in test mode

        Raises:
            ValueError: If title or body is invalid
            RuntimeError: If issue creation fails
        """
        # Input validation
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Title must be a non-empty string")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Body must be a non-empty string")
        if len(title) > Config.MAX_ISSUE_TITLE_LENGTH:
            raise ValueError(f"Title exceeds maximum length of {Config.MAX_ISSUE_TITLE_LENGTH} characters")
        if len(body) > Config.MAX_ISSUE_BODY_LENGTH:
            raise ValueError(f"Body exceeds maximum length of {Config.MAX_ISSUE_BODY_LENGTH} characters")

        if self.test_mode:
            logger.info(f"Would create issue: {title}")
            return None

        try:
            # Check for duplicate issues
            existing_issues = self.repo.get_issues(state="open")
            if any(issue.title.lower() == title.lower() for issue in existing_issues):
                logger.info(f"Issue already exists: {title}")
                return None

            # Create issue
            issue = self.repo.create_issue(title=title, body=body, labels=labels or [])
            logger.info(f"Created issue #{issue.number}: {title}")
            return issue

        except Exception as e:
            logger.error(f"Failed to create issue: {str(e)}")
            raise RuntimeError(f"Failed to create issue: {str(e)}")

    def tag_version(self, version: str, message: Optional[str] = None) -> None:
        """Create a git tag for a version.

        Args:
            version: Version string to tag
            message: Optional tag message

        Raises:
            ValueError: If version is invalid
            RuntimeError: If tagging fails
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version must be a non-empty string")

        if self.test_mode:
            logger.info(f"Would create tag: {version}")
            return

        try:
            # Get latest commit
            commits = self.repo.get_commits()
            latest_commit = commits[0]

            # Create tag
            tag_name = f"v{version}"
            self.repo.create_git_ref(ref=f"refs/tags/{tag_name}", sha=latest_commit.sha)

            if message:
                self.repo.create_git_tag(tag=tag_name, message=message, object=latest_commit.sha, type="commit")

            logger.info(f"Created tag {tag_name} at {latest_commit.sha[:8]}")

        except Exception as e:
            logger.error(f"Failed to create tag: {str(e)}")
            raise RuntimeError(f"Failed to create tag: {str(e)}")

    def rollback_to_version(self, version: str) -> None:
        """Roll back to a previous version.

        Args:
            version: Version to roll back to

        Raises:
            ValueError: If version is invalid
            RuntimeError: If rollback fails
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version must be a non-empty string")

        if self.test_mode:
            logger.info(f"Would roll back to version: {version}")
            return

        try:
            # Get tag
            tag_name = f"v{version}"
            tag = self.repo.get_git_ref(f"refs/tags/{tag_name}")

            # Get commit
            commit_sha = tag.object.sha
            commit = self.repo.get_commit(sha=commit_sha)

            # Reset main branch
            main_branch = self.repo.get_branch("main")
            main_branch.edit(sha=commit.sha)

            logger.info(f"Rolled back to version {version} (commit {commit.sha[:8]})")

        except Exception as e:
            logger.error(f"Failed to roll back: {str(e)}")
            raise RuntimeError(f"Failed to roll back: {str(e)}")

    def trigger_workflow(self, workflow_name: str, inputs: Optional[Dict[str, Any]] = None) -> None:
        """Trigger a GitHub Actions workflow.

        Args:
            workflow_name: Name of the workflow to trigger
            inputs: Optional workflow inputs

        Raises:
            ValueError: If workflow name is invalid
            RuntimeError: If workflow trigger fails
        """
        if not isinstance(workflow_name, str) or not workflow_name.strip():
            raise ValueError("Workflow name must be a non-empty string")

        if self.test_mode:
            logger.info(f"Would trigger workflow: {workflow_name}")
            return

        try:
            self.repo.create_repository_dispatch(workflow_name, {"event_type": "manual", "client_payload": inputs or {}})
            logger.info(f"Triggered workflow: {workflow_name}")

        except Exception as e:
            logger.error(f"Failed to trigger workflow: {str(e)}")
            raise RuntimeError(f"Failed to trigger workflow: {str(e)}")

    def get_commit_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent commit history.

        Args:
            days: Number of days of history to retrieve

        Returns:
            List of commit information dictionaries

        Raises:
            ValueError: If days is invalid
            RuntimeError: If history retrieval fails
        """
        if not isinstance(days, int) or days < 1:
            raise ValueError("Days must be a positive integer")

        if self.test_mode:
            return []

        try:
            commits = []
            for commit in self.repo.get_commits():
                commit_info = {
                    "sha": commit.sha,
                    "author": commit.author.login if commit.author else "unknown",
                    "date": commit.commit.author.date.isoformat(),
                    "message": commit.commit.message,
                    "files": [f.filename for f in commit.files],
                }
                commits.append(commit_info)

            return commits

        except Exception as e:
            logger.error(f"Failed to get commit history: {str(e)}")
            raise RuntimeError(f"Failed to get commit history: {str(e)}")

    def check_rate_limits(self) -> Dict[str, int]:
        """Check GitHub API rate limits.

        Returns:
            Dictionary with rate limit information

        Raises:
            RuntimeError: If rate limit check fails
        """
        if self.test_mode:
            return {"remaining": 5000, "limit": 5000, "reset_time": 3600}

        try:
            rate_limit = self.client.get_rate_limit()
            return {
                "remaining": rate_limit.core.remaining,
                "limit": rate_limit.core.limit,
                "reset_time": rate_limit.core.reset.timestamp(),
            }

        except Exception as e:
            logger.error(f"Failed to check rate limits: {str(e)}")
            raise RuntimeError(f"Failed to check rate limits: {str(e)}")

    def close_issue(self, issue_number: int, comment: Optional[str] = None) -> None:
        """Close a GitHub issue.

        Args:
            issue_number: Issue number to close
            comment: Optional closing comment

        Raises:
            ValueError: If issue number is invalid
            RuntimeError: If issue closure fails
        """
        if not isinstance(issue_number, int) or issue_number < 1:
            raise ValueError("Issue number must be a positive integer")

        if self.test_mode:
            logger.info(f"Would close issue #{issue_number}")
            return

        try:
            issue = self.repo.get_issue(issue_number)

            if comment:
                issue.create_comment(comment)

            issue.edit(state="closed")
            logger.info(f"Closed issue #{issue_number}")

        except Exception as e:
            logger.error(f"Failed to close issue: {str(e)}")
            raise RuntimeError(f"Failed to close issue: {str(e)}")

    def create_branch(self, branch_name: str, base_branch: str = "main") -> None:
        """Create a new branch.

        Args:
            branch_name: Name of the new branch
            base_branch: Branch to create from

        Raises:
            ValueError: If branch name is invalid
            RuntimeError: If branch creation fails
        """
        if not isinstance(branch_name, str) or not branch_name.strip():
            raise ValueError("Branch name must be a non-empty string")

        if self.test_mode:
            logger.info(f"Would create branch: {branch_name}")
            return

        try:
            # Get base branch's latest commit
            base = self.repo.get_branch(base_branch)

            # Create new branch
            self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base.commit.sha)
            logger.info(f"Created branch {branch_name} from {base_branch}")

        except Exception as e:
            logger.error(f"Failed to create branch: {str(e)}")
            raise RuntimeError(f"Failed to create branch: {str(e)}")

    def create_pull_request(self, title: str, body: str, head: str, base: str = "main", draft: bool = False) -> None:
        """Create a pull request.

        Args:
            title: PR title
            body: PR description
            head: Head branch
            base: Base branch
            draft: Whether to create as draft

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If PR creation fails
        """
        if not all(isinstance(x, str) and x.strip() for x in [title, body, head, base]):
            raise ValueError("All string parameters must be non-empty")

        if self.test_mode:
            logger.info(f"Would create PR: {title}")
            return

        try:
            self.repo.create_pull(title=title, body=body, head=head, base=base, draft=draft)
            logger.info(f"Created PR: {title}")

        except Exception as e:
            logger.error(f"Failed to create PR: {str(e)}")
            raise RuntimeError(f"Failed to create PR: {str(e)}")
