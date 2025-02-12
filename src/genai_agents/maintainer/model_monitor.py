"""Model monitoring module."""

import logging
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from returns.result import Result, Success, Failure
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Gauge, Summary

from ..config import Config
from .exceptions import ModelMonitoringError

# Define Prometheus metrics
MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy", ["version"])
MODEL_LATENCY = Summary("model_latency", "Model latency", ["version"])
MODEL_THROUGHPUT = Gauge("model_throughput", "Model throughput", ["version"])


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    accuracy: float
    latency: float
    throughput: int
    memory_usage: float
    error_rate: float
    timestamp: str


class ModelMonitor:
    """Handles model performance monitoring and metrics tracking."""

    def __init__(self, github_manager, logger: Optional[logging.Logger] = None):
        """Initialize the model monitor.

        Args:
            github_manager: GitHub manager for issue creation
            logger: Optional logger instance
        """
        self.github_manager = github_manager
        self.logger = logger or logging.getLogger(__name__)
        self.model_performance = {}
        self.retraining_thresholds = Config.get_model_thresholds()

    def record_metrics(self, version: str, metrics: ModelMetrics) -> Result[bool, Exception]:
        """Record model performance metrics.

        Args:
            version: Model version
            metrics: Performance metrics

        Returns:
            Result indicating success

        Raises:
            ModelMonitoringError: If recording fails
        """
        try:
            # Store metrics
            self.model_performance[version] = metrics

            # Update Prometheus metrics
            MODEL_ACCURACY.labels(version=version).set(metrics.accuracy)
            MODEL_LATENCY.labels(version=version).observe(metrics.latency)
            MODEL_THROUGHPUT.labels(version=version).set(metrics.throughput)

            # Save to state
            Config.save_state(
                Config.MODEL_METRICS_PATH,
                {
                    version: {
                        "accuracy": metrics.accuracy,
                        "latency": metrics.latency,
                        "throughput": metrics.throughput,
                        "memory_usage": metrics.memory_usage,
                        "error_rate": metrics.error_rate,
                        "timestamp": metrics.timestamp,
                    }
                },
            )

            # Check thresholds and create issues if needed
            self._check_performance_thresholds(version, metrics)

            return Success(True)

        except Exception as e:
            self.logger.exception("Failed to record metrics")
            return Failure(ModelMonitoringError(f"Failed to record metrics: {str(e)}"))

    def _check_performance_thresholds(self, version: str, metrics: ModelMetrics) -> None:
        """Check if model performance meets thresholds.

        Args:
            version: Model version
            metrics: Current metrics
        """
        issues = []

        # Check accuracy
        if metrics.accuracy < self.retraining_thresholds["accuracy"]:
            issues.append(
                {
                    "title": f"Model Accuracy Below Threshold (v{version})",
                    "body": f"Model accuracy ({metrics.accuracy:.2f}) is below threshold ({self.retraining_thresholds['accuracy']:.2f})",
                    "labels": ["model-performance", "accuracy"],
                }
            )

        # Check latency
        if metrics.latency > self.retraining_thresholds["latency"]:
            issues.append(
                {
                    "title": f"Model Latency Above Threshold (v{version})",
                    "body": f"Model latency ({metrics.latency:.2f}s) is above threshold ({self.retraining_thresholds['latency']:.2f}s)",
                    "labels": ["model-performance", "latency"],
                }
            )

        # Check throughput
        min_throughput = self.retraining_thresholds.get("min_throughput")
        if min_throughput and metrics.throughput < min_throughput:
            issues.append(
                {
                    "title": f"Model Throughput Below Threshold (v{version})",
                    "body": f"Model throughput ({metrics.throughput} req/s) is below threshold ({min_throughput} req/s)",
                    "labels": ["model-performance", "throughput"],
                }
            )

        # Check error rate
        max_error_rate = self.retraining_thresholds.get("max_error_rate", 0.01)
        if metrics.error_rate > max_error_rate:
            issues.append(
                {
                    "title": f"Model Error Rate Above Threshold (v{version})",
                    "body": f"Model error rate ({metrics.error_rate:.2%}) is above threshold ({max_error_rate:.2%})",
                    "labels": ["model-performance", "errors"],
                }
            )

        # Create GitHub issues
        for issue in issues:
            self.github_manager.create_issue(**issue)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def run_performance_test(self, model, test_data: List[Dict[str, Any]]) -> Result[ModelMetrics, Exception]:
        """Run comprehensive performance test.

        Args:
            model: Model to test
            test_data: Test dataset

        Returns:
            Result containing performance metrics

        Raises:
            ModelMonitoringError: If test fails
        """
        try:
            start_time = time.time()
            total_requests = len(test_data)
            successful_predictions = 0
            errors = 0
            latencies = []

            for item in test_data:
                try:
                    # Time the prediction
                    pred_start = time.time()
                    prediction = model.predict(item["input"])
                    latency = time.time() - pred_start
                    latencies.append(latency)

                    # Check accuracy
                    if prediction == item["expected"]:
                        successful_predictions += 1
                except Exception:
                    errors += 1
                    self.logger.exception("Prediction failed")

            # Calculate metrics
            duration = time.time() - start_time
            metrics = ModelMetrics(
                accuracy=successful_predictions / total_requests,
                latency=sum(latencies) / len(latencies) if latencies else float("inf"),
                throughput=int(total_requests / duration),
                memory_usage=self._get_memory_usage(model),
                error_rate=errors / total_requests,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            return Success(metrics)

        except Exception as e:
            self.logger.exception("Performance test failed")
            return Failure(ModelMonitoringError(f"Performance test failed: {str(e)}"))

    def _get_memory_usage(self, model) -> float:
        """Get model's memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            self.logger.exception("Failed to get memory usage")
            return 0.0

    def get_performance_history(self, version: str) -> Result[List[ModelMetrics], Exception]:
        """Get historical performance data.

        Args:
            version: Model version

        Returns:
            Result containing performance history

        Raises:
            ModelMonitoringError: If retrieval fails
        """
        try:
            history = Config.load_state(Config.MODEL_METRICS_PATH)
            if not history or version not in history:
                return Success([])

            metrics_list = []
            for data in history[version]:
                metrics_list.append(
                    ModelMetrics(
                        accuracy=data["accuracy"],
                        latency=data["latency"],
                        throughput=data["throughput"],
                        memory_usage=data["memory_usage"],
                        error_rate=data["error_rate"],
                        timestamp=data["timestamp"],
                    )
                )

            return Success(metrics_list)

        except Exception as e:
            self.logger.exception("Failed to get performance history")
            return Failure(ModelMonitoringError(f"Failed to get performance history: {str(e)}"))

    def compare_versions(self, version1: str, version2: str) -> Result[Dict[str, Any], Exception]:
        """Compare metrics between two versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Result containing comparison results

        Raises:
            ModelMonitoringError: If comparison fails
        """
        try:
            metrics1 = self.model_performance.get(version1)
            metrics2 = self.model_performance.get(version2)

            if not metrics1 or not metrics2:
                return Failure(ModelMonitoringError("Metrics not found for one or both versions"))

            comparison = {
                "accuracy": {
                    "old": metrics1.accuracy,
                    "new": metrics2.accuracy,
                    "change": metrics2.accuracy - metrics1.accuracy,
                    "percent_change": ((metrics2.accuracy - metrics1.accuracy) / metrics1.accuracy) * 100,
                },
                "latency": {
                    "old": metrics1.latency,
                    "new": metrics2.latency,
                    "change": metrics2.latency - metrics1.latency,
                    "percent_change": ((metrics2.latency - metrics1.latency) / metrics1.latency) * 100,
                },
                "throughput": {
                    "old": metrics1.throughput,
                    "new": metrics2.throughput,
                    "change": metrics2.throughput - metrics1.throughput,
                    "percent_change": ((metrics2.throughput - metrics1.throughput) / metrics1.throughput) * 100,
                },
            }

            # Create comparison report issue
            self._create_comparison_issue(version1, version2, comparison)

            return Success(comparison)

        except Exception as e:
            self.logger.exception("Version comparison failed")
            return Failure(ModelMonitoringError(f"Version comparison failed: {str(e)}"))

    def _create_comparison_issue(self, version1: str, version2: str, comparison: Dict[str, Any]) -> None:
        """Create GitHub issue for version comparison.

        Args:
            version1: First version
            version2: Second version
            comparison: Comparison results
        """
        body = f"""
        # Model Version Comparison
        
        Comparing v{version1} to v{version2}:
        
        ## Accuracy
        - Old: {comparison['accuracy']['old']:.2%}
        - New: {comparison['accuracy']['new']:.2%}
        - Change: {comparison['accuracy']['change']:.2%}
        - Percent Change: {comparison['accuracy']['percent_change']:.1f}%
        
        ## Latency
        - Old: {comparison['latency']['old']:.2f}s
        - New: {comparison['latency']['new']:.2f}s
        - Change: {comparison['latency']['change']:.2f}s
        - Percent Change: {comparison['latency']['percent_change']:.1f}%
        
        ## Throughput
        - Old: {comparison['throughput']['old']} req/s
        - New: {comparison['throughput']['new']} req/s
        - Change: {comparison['throughput']['change']} req/s
        - Percent Change: {comparison['throughput']['percent_change']:.1f}%
        """

        self.github_manager.create_issue(
            title=f"Model Version Comparison: v{version1} vs v{version2}",
            body=body,
            labels=["model-performance", "version-comparison"],
        )

    def check_drift(self, version: str, window_size: int = 7) -> Result[Dict[str, float], Exception]:
        """Check for model drift over time.

        Args:
            version: Model version
            window_size: Days to analyze

        Returns:
            Result containing drift metrics

        Raises:
            ModelMonitoringError: If analysis fails
        """
        try:
            history_result = self.get_performance_history(version)
            if history_result.is_failure():
                return history_result

            history = history_result.unwrap()
            if len(history) < window_size:
                return Success({})  # Not enough data

            # Calculate drift metrics
            recent = history[-window_size:]
            baseline = history[0]

            drift_metrics = {
                "accuracy_drift": recent[-1].accuracy - baseline.accuracy,
                "latency_drift": recent[-1].latency - baseline.latency,
                "error_rate_drift": recent[-1].error_rate - baseline.error_rate,
            }

            # Check for significant drift
            if abs(drift_metrics["accuracy_drift"]) > Config.DRIFT_THRESHOLDS["accuracy"]:
                self.github_manager.create_issue(
                    title=f"Model Accuracy Drift Detected (v{version})",
                    body=f"Accuracy has drifted by {drift_metrics['accuracy_drift']:.2%} over {window_size} days",
                    labels=["model-performance", "drift"],
                )

            return Success(drift_metrics)

        except Exception as e:
            self.logger.exception("Drift analysis failed")
            return Failure(ModelMonitoringError(f"Drift analysis failed: {str(e)}"))
