"""
Model Monitoring Module

Handles model performance monitoring and metrics tracking.
"""

import logging
from typing import Dict, Any, Optional, List
import time
from prometheus_client import Gauge, Summary
from ..config import Config

logger = logging.getLogger(__name__)

# Define Prometheus metrics
MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy", ["version"])
MODEL_LATENCY = Summary("model_latency", "Model latency", ["version"])
MODEL_THROUGHPUT = Gauge("model_throughput", "Model throughput", ["version"])


class ModelMonitor:
    """Handles model performance monitoring and metrics tracking"""

    def __init__(self, test_mode: bool = False):
        """Initialize the model monitor.

        Args:
            test_mode: Whether to use mock responses
        """
        self.test_mode = test_mode
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        self.retraining_thresholds = Config.get_model_thresholds()

    def record_metrics(self, version: str, metrics: Dict[str, float]) -> None:
        """Record model performance metrics.

        Args:
            version: Model version
            metrics: Dictionary of metric names to values

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version must be a non-empty string")
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")

        try:
            # Store metrics
            self.model_performance[version] = metrics

            # Update Prometheus metrics
            MODEL_ACCURACY.labels(version=version).set(metrics.get("accuracy", 0))
            MODEL_LATENCY.labels(version=version).observe(metrics.get("latency", 0))
            MODEL_THROUGHPUT.labels(version=version).set(metrics.get("throughput", 0))

            logger.info(f"Recorded metrics for version {version}: {metrics}")

        except Exception as e:
            logger.error(f"Failed to record metrics: {str(e)}")
            raise RuntimeError(f"Failed to record metrics: {str(e)}")

    def check_performance(self, version: str) -> Dict[str, Any]:
        """Check if model performance meets thresholds.

        Args:
            version: Model version to check

        Returns:
            Dictionary with performance status and details

        Raises:
            ValueError: If version is invalid
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version must be a non-empty string")

        if version not in self.model_performance:
            raise ValueError(f"No metrics found for version {version}")

        try:
            metrics = self.model_performance[version]
            thresholds = self.retraining_thresholds

            status = {"needs_retraining": False, "reasons": [], "metrics": metrics}

            # Check accuracy
            if metrics.get("accuracy", 1.0) < thresholds["accuracy"]:
                status["needs_retraining"] = True
                status["reasons"].append(f"Accuracy {metrics['accuracy']:.2f} below threshold {thresholds['accuracy']:.2f}")

            # Check latency
            if metrics.get("latency", 0.0) > thresholds["latency"]:
                status["needs_retraining"] = True
                status["reasons"].append(f"Latency {metrics['latency']:.2f}s above threshold {thresholds['latency']:.2f}s")

            # Check throughput
            min_throughput = thresholds.get("min_throughput")
            if min_throughput and metrics.get("throughput", float("inf")) < min_throughput:
                status["needs_retraining"] = True
                status["reasons"].append(f"Throughput {metrics['throughput']} below threshold {min_throughput}")

            return status

        except Exception as e:
            logger.error(f"Failed to check performance: {str(e)}")
            raise RuntimeError(f"Failed to check performance: {str(e)}")

    def get_version_metrics(self, version: str) -> Optional[Dict[str, float]]:
        """Get metrics for a specific version.

        Args:
            version: Model version

        Returns:
            Dictionary of metrics or None if not found

        Raises:
            ValueError: If version is invalid
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version must be a non-empty string")

        return self.model_performance.get(version)

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all versions.

        Returns:
            Dictionary mapping versions to their metrics
        """
        return self.model_performance.copy()

    def run_accuracy_test(self, model, test_data) -> float:
        """Run accuracy test on model.

        Args:
            model: Model to test
            test_data: Test dataset

        Returns:
            Accuracy score

        Raises:
            RuntimeError: If test fails
        """
        if self.test_mode:
            return 0.95

        try:
            # Run predictions
            correct = 0
            total = len(test_data)

            start_time = time.time()
            for input_data, expected in test_data:
                prediction = model.predict(input_data)
                if prediction == expected:
                    correct += 1
            end_time = time.time()

            # Record latency
            latency = (end_time - start_time) / total
            self.record_metrics(model.version, {"latency": latency})

            return correct / total

        except Exception as e:
            logger.error(f"Failed to run accuracy test: {str(e)}")
            raise RuntimeError(f"Failed to run accuracy test: {str(e)}")

    def measure_inference_speed(self, model, test_data) -> float:
        """Measure model inference speed.

        Args:
            model: Model to test
            test_data: Test dataset

        Returns:
            Average latency in seconds

        Raises:
            RuntimeError: If measurement fails
        """
        if self.test_mode:
            return 1.5

        try:
            latencies = []
            for input_data, _ in test_data:
                start_time = time.time()
                model.predict(input_data)
                end_time = time.time()
                latencies.append(end_time - start_time)

            return sum(latencies) / len(latencies)

        except Exception as e:
            logger.error(f"Failed to measure inference speed: {str(e)}")
            raise RuntimeError(f"Failed to measure inference speed: {str(e)}")

    def run_stress_test(self, model, test_data, duration: int = 60) -> int:
        """Run stress test to measure throughput.

        Args:
            model: Model to test
            test_data: Test dataset
            duration: Test duration in seconds

        Returns:
            Requests per second

        Raises:
            ValueError: If duration is invalid
            RuntimeError: If test fails
        """
        if not isinstance(duration, int) or duration < 1:
            raise ValueError("Duration must be a positive integer")

        if self.test_mode:
            return 1000

        try:
            start_time = time.time()
            requests = 0

            while time.time() - start_time < duration:
                for input_data, _ in test_data:
                    model.predict(input_data)
                    requests += 1

            elapsed = time.time() - start_time
            throughput = int(requests / elapsed)

            # Record throughput
            self.record_metrics(model.version, {"throughput": throughput})

            return throughput

        except Exception as e:
            logger.error(f"Failed to run stress test: {str(e)}")
            raise RuntimeError(f"Failed to run stress test: {str(e)}")

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare metrics between two versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Dictionary with comparison results

        Raises:
            ValueError: If versions are invalid
        """
        if not all(isinstance(v, str) and v.strip() for v in [version1, version2]):
            raise ValueError("Versions must be non-empty strings")

        metrics1 = self.get_version_metrics(version1)
        metrics2 = self.get_version_metrics(version2)

        if not metrics1 or not metrics2:
            raise ValueError("Metrics not found for one or both versions")

        try:
            comparison = {}
            for metric in set(metrics1.keys()) & set(metrics2.keys()):
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                pct_change = ((val2 - val1) / val1) * 100

                comparison[metric] = {"old": val1, "new": val2, "change": val2 - val1, "percent_change": pct_change}

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare versions: {str(e)}")
            raise RuntimeError(f"Failed to compare versions: {str(e)}")

    def get_performance_history(self, version: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical performance data.

        Args:
            version: Model version
            days: Number of days of history

        Returns:
            List of historical metrics

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Version must be a non-empty string")
        if not isinstance(days, int) or days < 1:
            raise ValueError("Days must be a positive integer")

        if self.test_mode:
            return []

        try:
            # This would typically query a time series database
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Failed to get performance history: {str(e)}")
            raise RuntimeError(f"Failed to get performance history: {str(e)}")
