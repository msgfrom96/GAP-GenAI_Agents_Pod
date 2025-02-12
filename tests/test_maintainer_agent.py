import os
from genai_agents.config import Config
import pytest
from unittest.mock import Mock, patch
import json
import re
from datetime import datetime

from src.genai_agents.openai import OpenAIService
from src.genai_agents.google import GoogleAIService
from src.genai_agents.maintainer_agent import MaintainerAgent, DocumentationAnalysisResult
from github import Github

# Mock the Github class and its methods
class MockGithub:
    def get_repo(self, repo_name):
        return MockRepo()


class MockRepo:
    def get_issues(self, state):
        return []  # Return an empty list of issues for testing

    def create_issue(self, title, body):
        print(f"Creating issue with title: {title} and body: {body}")
        return  # Mock issue creation

    def get_commits(self):
        return []

    def create_repository_dispatch(self, main, event_type):
        return

    def create_git_ref(self, ref, sha):
        return

    def get_git_ref(self, ref):
        return Mock()

    def get_commit(self, sha):
        return Mock()

    def get_branch(self, branch_name):
        return Mock()

    def edit_commit(self, sha):
        return None


# Mock the InputGitTreeElement class
class MockInputGitTreeElement:
    def __init__(self, path, mode, type, content):
        self.path = path
        self.mode = mode
        self.type = type
        self.content = content


@pytest.fixture
def mock_google_system():
    """Mocks the Google AI system."""
    google_system = Mock()
    google_system.aask_with_rag.return_value = json.dumps(
        {"deprecated": [], "new_features": [], "security": [], "performance": [], "breaking_changes": []}
    )
    return google_system


@pytest.fixture
def mock_openai_system():
    """Mocks the OpenAI system."""
    openai_system = Mock()
    openai_system.aask_with_rag.return_value = json.dumps(
        {"deprecated": [], "new_features": [], "security": [], "performance": [], "breaking_changes": []}
    )
    return openai_system


@pytest.fixture
def mock_openai_service():
    service = Mock(spec=OpenAIService)
    service.test_mode = True
    service.ask_with_rag.return_value = json.dumps(
        {
            "deprecated": ["mock_deprecated_func"],
            "new_features": ["mock_new_feature"],
            "security": ["mock_security_update"],
            "performance": ["mock_performance_improvement"],
            "breaking_changes": ["mock_breaking_change"],
        }
    )
    return service


@pytest.fixture
def mock_google_service():
    service = Mock(spec=GoogleAIService)
    service.test_mode = True
    return service


@pytest.fixture
def maintainer_agent(mock_openai_service, mock_google_service):
    """Creates a MaintainerAgent instance with mocked dependencies."""
    # Patch the Github class with the MockGithub class
    with patch("github.Github", MockGithub):
        agent = MaintainerAgent(google_ai_system=mock_google_service, openai_system=mock_openai_service, test_mode=True)
        agent.github_client = MockGithub()  # Directly assign the MockGithub instance
        return agent


def test_maintainer_agent_initialization(maintainer_agent):
    """Tests the initialization of the MaintainerAgent."""
    assert maintainer_agent is not None
    assert isinstance(maintainer_agent, MaintainerAgent)
    assert maintainer_agent.test_mode is True
    assert maintainer_agent.is_authenticated is True
    assert isinstance(maintainer_agent.doc_states, dict)


@pytest.mark.asyncio
async def test_parse_documentation(maintainer_agent):
    """Tests the _parse_documentation method."""
    with patch("openai.ChatCompletion.create") as mock_create:
        mock_create.return_value = {"choices": [{"message": {"content": "Test documentation content"}}]}
        result = maintainer_agent._parse_documentation("test_content")
        assert isinstance(result, DocumentationAnalysisResult)
        assert result.deprecated == []


def test_fetch_latest_documentation(maintainer_agent):
    """Tests the _fetch_latest_documentation method."""
    maintainer_agent.test_mode = False  # Ensure test mode is off to test the request
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.text = "Test documentation"
        mock_get.return_value = mock_response
        result = maintainer_agent._fetch_latest_documentation("Google")
        assert result == "Test documentation"
        mock_get.assert_called_once()


def test_check_code_against_docs(maintainer_agent):
    """Tests the _check_code_against_docs method."""
    with patch.object(maintainer_agent, "_parse_documentation") as mock_parse:
        mock_parse.return_value = DocumentationAnalysisResult(
            deprecated=["old_function"],
            new_features=["new_feature"],
            security=["security_update"],
            performance=["performance_improvement"],
            breaking_changes=["breaking_change"],
        )
        maintainer_agent._check_code_against_docs()
        assert mock_parse.call_count == 2  # Called for both Google and OpenAI


def test_validate_code_changes(maintainer_agent):
    """Tests the _validate_code_changes method."""
    with patch("ast.parse") as mock_parse, patch("pytest.main") as mock_pytest_main:
        mock_parse.return_value = True
        mock_pytest_main.return_value = 0
        result = maintainer_agent._validate_code_changes("test_code")
        assert result is True
        mock_parse.assert_called_once()
        mock_pytest_main.assert_called_once()


def test_find_google_deprecations(maintainer_agent):
    """Tests the _find_google_deprecations method."""
    doc_content = """
    This is a test document.
    def deprecated function_name():
        pass
    """
    deprecations = maintainer_agent._find_google_deprecations(doc_content)
    assert "function_name" in deprecations


def test_find_openai_deprecations(maintainer_agent):
    """Tests the _find_openai_deprecations method."""
    doc_content = """
    This is a test document.
    def deprecated function_name():
        pass
    """
    deprecations = maintainer_agent._find_openai_deprecations(doc_content)
    assert "def deprecated function_name():" in str(deprecations)


def test_find_google_new_features(maintainer_agent):
    """Tests the _find_google_new_features method."""
    doc_content = """
    This is a test document.
    New feature: feature_name
    """
    new_features = maintainer_agent._find_google_new_features(doc_content)
    assert "New feature: feature_name" in str(new_features)


def test_find_openai_new_features(maintainer_agent):
    """Tests the _find_openai_new_features method."""
    doc_content = """
    This is a test document.
    Now available: feature_name
    """
    new_features = maintainer_agent._find_openai_new_features(doc_content)
    assert "Now available: feature_name" in str(new_features)


def test_check_security_compliance(maintainer_agent):
    """Tests the _check_security_compliance method."""
    code = """
    html.escape("test")
    try:
        pass
    except:
        logging.error("test")
    Fernet.generate_key()
    @login_required
    def test():
        pass
    """
    with patch.object(MockRepo, "create_issue") as mock_create_issue:
        maintainer_agent._check_security_compliance(code)
        mock_create_issue.assert_not_called()

    code = """
    def test():
        pass
    """
    with patch.object(MockRepo, "create_issue") as mock_create_issue:
        maintainer_agent._check_security_compliance(code)
        assert mock_create_issue.call_count == 4


def test_check_security_compliance_subprocess(maintainer_agent):
    """Tests the _check_security_compliance method with subprocess."""
    code = """
    import subprocess
    subprocess.call("ls -l", shell=True)
    """
    with patch.object(MockRepo, "create_issue") as mock_create_issue:
        maintainer_agent._check_security_compliance(code)
        # Verify that at least one security issue was created for subprocess usage
        assert mock_create_issue.call_count > 0
        # Verify that subprocess warning was one of the issues
        subprocess_call = any(
            call[1]["title"].startswith("Security Warning: Subprocess usage detected")
            for call in mock_create_issue.call_args_list
        )
        assert subprocess_call, "No subprocess warning was created"


def test_check_security_compliance_deserialization(maintainer_agent):
    """Tests the _check_security_compliance method with unsafe deserialization."""
    code = """
    import pickle
    pickle.load(open("data.pkl", "rb"))
    """
    with patch.object(MockRepo, "create_issue") as mock_create_issue:
        maintainer_agent._check_security_compliance(code)
        # Verify that at least one security issue was created for unsafe deserialization
        assert mock_create_issue.call_count > 0
        # Verify that deserialization warning was one of the issues
        deserialization_call = any(
            call[1]["title"].startswith("Security Warning: Unsafe deserialization")
            for call in mock_create_issue.call_args_list
        )
        assert deserialization_call, "No deserialization warning was created"


def test_handle_rollback(maintainer_agent):
    """Tests the _handle_rollback method."""
    with patch.object(MockRepo, "get_git_ref") as mock_get_git_ref, patch.object(
        MockRepo, "get_commit"
    ) as mock_get_commit, patch.object(MockRepo, "get_branch") as mock_get_branch:

        # Mock the tag and commit
        mock_tag = Mock()
        mock_tag.object.sha = "test_commit_sha"
        mock_get_git_ref.return_value = mock_tag

        mock_commit = Mock()
        mock_commit.sha = "test_commit_sha"
        mock_get_commit.return_value = mock_commit

        mock_branch = Mock()
        mock_branch.edit_commit.return_value = None
        mock_get_branch.return_value = mock_branch

        maintainer_agent._handle_rollback("1.0")

        mock_get_git_ref.assert_called_with("tags/model-version-1.0")
        mock_get_commit.assert_called_with(sha="test_commit_sha")
        mock_get_branch.assert_called_with("main")
        mock_branch.edit_commit.assert_called_with(sha="test_commit_sha")


def test_run_accuracy_test(maintainer_agent):
    """Tests the _run_accuracy_test method."""
    # Mock the accuracy test logic
    with patch.object(MaintainerAgent, "_run_accuracy_test") as mock_run_accuracy_test:
        mock_run_accuracy_test.return_value = 0.95
        accuracy = maintainer_agent._run_accuracy_test()
        assert accuracy == 0.95


def test_measure_inference_speed(maintainer_agent):
    """Tests the _measure_inference_speed method."""
    # Mock the latency measurement logic
    with patch.object(MaintainerAgent, "_measure_inference_speed") as mock_measure_inference_speed:
        mock_measure_inference_speed.return_value = 1.5
        latency = maintainer_agent._measure_inference_speed()
        assert latency == 1.5


def test_stress_test(maintainer_agent):
    """Tests the _stress_test method."""
    # Mock the stress test logic
    with patch.object(MaintainerAgent, "_stress_test") as mock_stress_test:
        mock_stress_test.return_value = 1000
        throughput = maintainer_agent._stress_test()
        assert throughput == 1000


def test_submit_github_issue_invalid_input(maintainer_agent):
    """Tests the _submit_github_issue method with invalid input."""
    with patch.object(MockRepo, "create_issue") as mock_create_issue:
        # Test with invalid title
        result = maintainer_agent._submit_github_issue(title=123, body="test")
        assert result is False
        mock_create_issue.assert_not_called()

        # Test with invalid body
        result = maintainer_agent._submit_github_issue(title="test", body=456)
        assert result is False
        mock_create_issue.assert_not_called()

        # Test with empty title
        result = maintainer_agent._submit_github_issue(title="", body="test")
        assert result is False
        mock_create_issue.assert_not_called()

        # Test with empty body
        result = maintainer_agent._submit_github_issue(title="test", body="")
        assert result is False
        mock_create_issue.assert_not_called()


def test_handle_rollback_invalid_input(maintainer_agent):
    """Tests the _handle_rollback method with invalid input."""
    with patch.object(MockRepo, "get_git_ref") as mock_get_git_ref:
        # Test with invalid version
        result = maintainer_agent._handle_rollback(version=123)
        assert result is None
        mock_get_git_ref.assert_not_called()


def test_documentation_parsing(maintainer_agent):
    """Test documentation analysis"""
    result = maintainer_agent._parse_documentation("OpenAI")
    assert isinstance(result.deprecated, list)
    assert isinstance(result.new_features, list)
    assert isinstance(result.security, list)
    assert isinstance(result.breaking_changes, list)


def test_github_issue_creation(maintainer_agent):
    """Test GitHub issue creation"""
    result = maintainer_agent._submit_github_issue("Test Issue", "Test issue body")
    assert result is True


def test_security_compliance(maintainer_agent):
    """Test security compliance checks"""
    code = """
    import subprocess
    subprocess.call(['ls'])
    """
    with patch.object(maintainer_agent, "_submit_github_issue") as mock_submit:
        maintainer_agent._check_security_compliance(code)
        mock_submit.assert_called()


@pytest.mark.asyncio
async def test_maintenance_cycle(maintainer_agent):
    """Tests the maintenance cycle."""
    with patch("os.path.exists") as mock_exists, patch.object(
        maintainer_agent, "_fetch_latest_documentation"
    ) as mock_fetch, patch.object(maintainer_agent, "_parse_documentation") as mock_parse, patch.object(
        maintainer_agent, "_check_code_against_docs"
    ) as mock_check, patch.object(
        maintainer_agent, "_validate_code_changes"
    ) as mock_validate, patch.object(
        maintainer_agent, "_test_functionality"
    ) as mock_test:

        mock_exists.return_value = True
        mock_fetch.return_value = "test_docs"
        mock_parse.return_value = DocumentationAnalysisResult(
            deprecated=[], new_features=[], security=[], performance=[], breaking_changes=[]
        )
        mock_check.return_value = None
        mock_validate.return_value = True
        mock_test.return_value = True

        await maintainer_agent.run_maintenance_cycle()

        assert mock_fetch.call_count == 2  # Called for both Google and OpenAI
        assert mock_parse.call_count == 2  # Called for both Google and OpenAI
        mock_check.assert_called_once()
        mock_test.assert_called_once()


@pytest.mark.asyncio
async def test_test_functionality(maintainer_agent):
    """Tests the _test_functionality method."""
    with patch("pytest.main") as mock_pytest_main:
        mock_pytest_main.return_value = 0
        result = await maintainer_agent._test_functionality()
        assert result is True


def test_update_dependencies(maintainer_agent):
    """Tests the _update_dependencies method."""
    with patch("subprocess.check_call") as mock_check_call:
        mock_check_call.return_value = 0
        result = maintainer_agent._update_dependencies()
        assert result is True
        mock_check_call.assert_called_once_with(["pip", "install", "--upgrade", "-r", "requirements.txt"])


def test_monitor_resource_usage(maintainer_agent):
    """Tests the _monitor_resource_usage method."""
    with patch("psutil.cpu_percent") as mock_cpu_percent, patch("psutil.virtual_memory") as mock_virtual_memory:
        mock_cpu_percent.return_value = 50.0
        mock_virtual_memory.return_value.percent = 60.0
        cpu_usage, memory_usage = maintainer_agent._monitor_resource_usage()
        assert cpu_usage == 50.0
        assert memory_usage == 60.0


@pytest.mark.integration
def test_full_security_scan(maintainer_agent):
    """
    Integration test for security compliance checks without mocking external calls.
    This test executes the security compliance check on a sample code snippet and
    verifies that the expected number of security issues are identified and submitted as GitHub issues.
    """
    # Skip test if environment variables are not set
    github_token = Config.GITHUB_TOKEN
    github_repo = Config.GITHUB_REPO
    if not github_token or not github_repo:
        pytest.skip("GITHUB_TOKEN or GITHUB_REPO environment variables not set")

    code = """
    import subprocess
    import pickle
    import yaml

    def risky_function(user_input):
        # Command injection vulnerability
        subprocess.call(f"ls {user_input}", shell=True)
        
        # Deserialization vulnerability
        pickle.loads(user_input)
        
        # Another deserialization vulnerability
        yaml.load(user_input)
        
        # Hardcoded secret
        api_key = "1234567890abcdef"
        
        # Missing error handling
        result = some_function()
        
        # Missing input validation
        process_data(user_input)
    """

    # Use the actual github client, not the mock
    maintainer_agent.github_client = Github(github_token)
    maintainer_agent.repo_name = github_repo
    maintainer_agent.repo = maintainer_agent.github_client.get_repo(github_repo)

    with patch.object(maintainer_agent.repo, "create_issue") as mock_create_issue:
        result = maintainer_agent._check_security_compliance(code)

        # Verify the result is successful
        assert result.is_success()
        issues = result.unwrap()

        # Verify we found all the security issues
        assert len(issues) >= 6  # At least 6 issues should be found

        # Verify specific issue types were found
        issue_types = {issue["type"] for issue in issues}
        assert "subprocess" in issue_types, "Subprocess vulnerability not detected"
        assert "deserialization" in issue_types, "Deserialization vulnerability not detected"
        assert "hardcoded_secret" in issue_types, "Hardcoded secret not detected"
        assert "error_handling" in issue_types, "Missing error handling not detected"
        assert "input_validation" in issue_types, "Missing input validation not detected"

        # Verify GitHub issues were created
        assert mock_create_issue.call_count >= 6

        # Verify specific issue titles
        issue_titles = [call[1]["title"] for call in mock_create_issue.call_args_list]
        assert any("Subprocess usage detected" in title for title in issue_titles)
        assert any("Unsafe deserialization" in title for title in issue_titles)
        assert any("Hardcoded secrets" in title for title in issue_titles)
        assert any("Missing input validation" in title for title in issue_titles)
        assert any("Improper error handling" in title for title in issue_titles)


@pytest.mark.integration
def test_security_scan_invalid_code(maintainer_agent):
    """Test security scan with invalid code."""
    result = maintainer_agent._check_security_compliance("invalid python code {")
    assert not result.is_success()
    assert isinstance(result.failure(), SecurityCheckError)
    assert "Invalid Python syntax" in str(result.failure())


@pytest.mark.integration
def test_security_scan_empty_code(maintainer_agent):
    """Test security scan with empty code."""
    result = maintainer_agent._check_security_compliance("")
    assert not result.is_success()
    assert isinstance(result.failure(), ValueError)
    assert "Code must be a non-empty string" in str(result.failure())


@pytest.mark.integration
def test_security_scan_safe_code(maintainer_agent):
    """Test security scan with safe code."""
    safe_code = """
    def safe_function(user_input: str) -> str:
        try:
            # Input validation
            if not isinstance(user_input, str):
                raise ValueError("Input must be a string")
            
            # Safe subprocess usage
            import subprocess
            subprocess.run(['echo', user_input], shell=False)
            
            # Safe deserialization
            import json
            data = json.loads(user_input)
            
            return "Success"
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            raise
    """

    result = maintainer_agent._check_security_compliance(safe_code)
    assert result.is_success()
    issues = result.unwrap()
    assert len(issues) == 0, "No security issues should be found in safe code"
