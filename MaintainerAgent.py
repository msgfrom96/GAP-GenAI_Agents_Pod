import os
import logging
import time
import requests
import inspect
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from github import Github, InputGitTreeElement
import pytest
import ast
import re
from collections import defaultdict

class MaintainerAgent:
    """AI-powered code maintenance agent with MLOps capabilities"""
    
    def __init__(self, google_ai_system, openai_system):
        self.google_system = google_ai_system
        self.openai_system = openai_system
        self.scheduler = BackgroundScheduler()
        self.github_client = Github(os.getenv('GITHUB_ACCESS_TOKEN'))
        self.last_check = None
        self.deprecated_functions = set()
        self.new_features_queue = []
        self.model_performance = {}  # Track accuracy/latency
        self.retraining_thresholds = {'accuracy': 0.85, 'latency': 2.0}
        self.versions = {
            'google_ai': '1.8.2',
            'openai': '1.12.0',
            'tensorflow': '2.16.1'
        }
        
        # Configure logging
        self.logger = logging.getLogger('MaintainerAgent')
        self.logger.setLevel(logging.INFO)
        
        if not os.getenv('GITHUB_ACCESS_TOKEN'):
            raise ValueError("Missing GITHUB_ACCESS_TOKEN in environment variables")

    def _parse_documentation(self, api_name: str) -> dict:
        """Enhanced doc analysis with AI-assisted parsing"""
        doc_content = self._fetch_latest_documentation(api_name)
        
        analysis_prompt = f"""
        Analyze this {api_name} API documentation and identify:
        1. Deprecated functions/methods with removal timelines
        2. New features with version requirements
        3. Security recommendations
        4. Performance improvements
        5. Breaking changes
        
        Return JSON format with keys: deprecated, new_features, security, performance, breaking_changes
        """
        
        response = self.openai_system.ask_with_rag(analysis_prompt, doc_content)
        return json.loads(response)

    def _find_google_deprecations(self, doc_content: str) -> list:
        """Identify deprecated Google AI functions"""
        return [
            line.split(' ')[1] for line in doc_content.split('\n') 
            if 'deprecated' in line.lower() and 'def ' in line
        ]

    def _find_openai_deprecations(self, doc_content: str) -> list:
        """Identify deprecated OpenAI functions"""
        return [
            line for line in doc_content.split('\n') 
            if 'deprecated' in line.lower() and 'def ' in line
        ]

    def _find_google_new_features(self, doc_content: str) -> list:
        """Detect new Google AI features"""
        return [
            line.strip() for line in doc_content.split('\n')
            if 'new feature' in line.lower() or 'introduced' in line.lower()
        ]

    def _find_openai_new_features(self, doc_content: str) -> list:
        """Detect new OpenAI features"""
        return [
            line.strip() for line in doc_content.split('\n')
            if 'new feature' in line.lower() or 'now available' in line.lower()
        ]

    def _fetch_latest_documentation(self, api_name: str) -> str:
        """Retrieve latest API documentation"""
        try:
            if api_name == 'Google':
                response = requests.get('https://ai.google.dev/docs')
            elif api_name == 'OpenAI':
                response = requests.get('https://api.openai.com/docs')
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"Failed to fetch {api_name} docs: {str(e)}")
            return ""

    def _check_code_against_docs(self):
        """Compare current code implementation with documentation"""
        google_changes = self._parse_documentation('Google')
        openai_changes = self._parse_documentation('OpenAI')
        
        # Check Google implementation
        current_google_funcs = set(
            name for name, _ in inspect.getmembers(self.google_system, inspect.ismethod)
        )
        self.deprecated_functions.update(
            func for func in google_changes['deprecated'] if func in current_google_funcs
        )
        
        # Check OpenAI implementation
        current_openai_funcs = set(
            name for name, _ in inspect.getmembers(self.openai_system, inspect.ismethod)
        )
        self.deprecated_functions.update(
            func for func in openai_changes['deprecated'] if func in current_openai_funcs
        )
        
        # Queue new features
        self.new_features_queue.extend(google_changes['new_features'])
        self.new_features_queue.extend(openai_changes['new_features'])

    def _submit_github_issue(self, title: str, body: str) -> bool:
        """Create GitHub issue for required changes"""
        try:
            repo = self.github_client.get_repo("your-org/your-repo")
            existing_issues = repo.get_issues(state='open')
            
            # Check for existing similar issues
            if any(title.lower() == issue.title.lower() for issue in existing_issues):
                self.logger.info(f"Issue '{title}' already exists")
                return False
                
            repo.create_issue(title=title, body=body)
            self.logger.info(f"Created new issue: {title}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create GitHub issue: {str(e)}")
            return False

    def _validate_code_changes(self, new_code: str) -> bool:
        """Validate generated code with AST parsing and test suite"""
        try:
            ast.parse(new_code)
            test_result = pytest.main(["--tb=line", "-k", "test_code_generation"])
            return test_result == 0
        except SyntaxError as e:
            self.logger.error(f"Invalid syntax: {str(e)}")
            return False

    def _test_functionality(self):
        """Run comprehensive test suite"""
        test_results = {
            'unit_tests': pytest.main(['-m', 'unit']),
            'integration_tests': pytest.main(['-m', 'integration']),
            'load_tests': pytest.main(['-m', 'load'])
        }
        
        if any(result != 0 for result in test_results.values()):
            raise RuntimeError("Test failures detected")
            
        self.logger.info("All tests passed successfully")

    def _update_documentation(self):
        """Update project documentation with new features"""
        try:
            with open('GenAI Agents/Google.py', 'r+') as f:
                content = f.read()
                if '# New Features' not in content:
                    f.write(f"\n# New Features\n# {datetime.now().isoformat()}\n")
                    for feature in self.new_features_queue:
                        f.write(f"# - {feature}\n")
        except Exception as e:
            self.logger.error(f"Documentation update failed: {str(e)}")

    def _perform_code_review(self):
        """Analyze code for deprecated patterns and potential issues"""
        # Check for deprecated function usage
        repo = self.github_client.get_repo("your-org/your-repo")
        commits = repo.get_commits()
        
        for commit in commits:
            files = commit.files
            for file in files:
                if file.filename.endswith('.py'):
                    patch = file.patch
                    for deprecated_func in self.deprecated_functions:
                        if deprecated_func in patch:
                            self._submit_github_issue(
                                title=f"Deprecated function {deprecated_func} used",
                                body=f"File: {file.filename}\nCommit: {commit.sha}"
                            )
                    self._check_security_compliance(patch)

    def _check_security_compliance(self, code: str):
        """Verify security best practices"""
        checks = [
            ("input sanitization", r"(html\.escape|bleach|sanitize)"),
            ("error handling", r"(try:|except |logging\.error)"),
            ("data encryption", r"(Fernet|AES|encrypt)"),
            ("auth checks", r"(@login_required|@permission_required)")
        ]
        
        for check_name, pattern in checks:
            if not re.search(pattern, code):
                self._submit_github_issue(
                    f"Security Gap: Missing {check_name}",
                    f"Found in code:\n{code}"
                )

    @staticmethod
    def _schedule_weekly_maintenance():
        """Configure weekly maintenance schedule"""
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            MaintainerAgent.run_maintenance_cycle,
            'interval',
            weeks=1,
            next_run_time=datetime.now()
        )
        scheduler.start()

    def run_maintenance_cycle(self):
        """Full maintenance workflow"""
        self.logger.info("Starting maintenance cycle")
        start_time = time.time()
        
        try:
            # Phase 1: Documentation checks
            self._check_code_against_docs()
            
            # Phase 2: Code analysis
            self._perform_code_review()
            
            # Phase 3: Test current implementation
            self._test_functionality()
            
            # Phase 4: Update documentation
            self._update_documentation()
            
            # Phase 5: Submit feature requests
            for feature in self.new_features_queue:
                self._submit_github_issue(
                    title=f"New Feature Request: {feature}",
                    body=f"Automated request for implementing {feature}"
                )
            
            self._monitor_model_performance()
            self._check_retraining_needs()
            
            self.logger.info(f"Maintenance cycle completed in {time.time()-start_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Maintenance cycle failed: {str(e)}")
            raise

    def _monitor_model_performance(self):
        """Track model metrics across versions"""
        current_stats = {
            'accuracy': self._run_accuracy_test(),
            'latency': self._measure_inference_speed(),
            'throughput': self._stress_test()
        }
        self.model_performance[self.active_model_version] = current_stats

    def _check_retraining_needs(self):
        """Evaluate if retraining is needed"""
        current = self.model_performance.get(self.active_model_version, {})
        if (current.get('accuracy', 1.0) < self.retraining_thresholds['accuracy'] or
            current.get('latency', 0.0) > self.retraining_thresholds['latency']):
            self._submit_github_issue(
                "Model Retraining Recommended", 
                f"Performance degradation detected: {current}"
            )

    def start(self):
        """Start scheduled maintenance"""
        self.scheduler.add_job(
            self.run_maintenance_cycle,
            'interval',
            weeks=1,
            next_run_time=datetime.now()
        )
        self.scheduler.start()
        self.logger.info("Maintainer Agent started with weekly schedule")

    def stop(self):
        """Stop maintenance activities"""
        self.scheduler.shutdown()
        self.logger.info("Maintainer Agent stopped")

    def _analyze_dependencies(self):
        """Map function dependencies across codebase"""
        dependency_graph = defaultdict(set)
        for fn in self._get_all_functions():
            calls = self._find_function_calls(fn)
            dependency_graph[fn].update(calls)
        return dependency_graph
    
    def _verify_test_coverage(self):
        """Ensure new features have tests"""
        coverage = pytest_cov.load()
        untested = [
            fn for fn in self.new_features_queue
            if not coverage.analysis(fn)[1]
        ]
        if untested:
            self._submit_github_issue(
                "Missing Test Coverage",
                f"Untested features: {', '.join(untested)}"
            )

    def _trigger_ci_pipeline(self):
        """Trigger CI/CD pipeline through GitHub API"""
        repo = self.github_client.get_repo("your-org/your-repo")
        repo.create_repository_dispatch("main", {"event_type": "mlops-pipeline"})

    def _handle_rollback(self, version: str):
        """Rollback model version using git tags"""
        repo = self.github_client.get_repo("your-org/your-repo")
        commit = repo.get_commit(sha=version)
        repo.create_git_ref(ref=f"refs/tags/rollback-{version}", sha=commit.sha)