"""Custom exceptions for the maintainer package."""


class MaintainerError(Exception):
    """Base class for maintainer exceptions."""

    pass


class SecurityCheckError(MaintainerError):
    """Raised when a security check fails."""

    pass


class DynamicAnalysisError(MaintainerError):
    """Raised when dynamic analysis fails."""

    pass


class SandboxError(MaintainerError):
    """Raised when sandbox execution fails."""

    pass


class StateError(MaintainerError):
    """Raised when state management fails."""

    pass


class ConfigurationError(MaintainerError):
    """Raised when configuration is invalid."""

    pass


class ValidationError(MaintainerError):
    """Raised when input validation fails."""

    pass


class DocumentationError(MaintainerError):
    """Raised when documentation analysis fails."""

    pass


class PricingError(MaintainerError):
    """Raised when pricing updates fail."""

    pass


class IssueTrackingError(MaintainerError):
    """Raised when GitHub issue tracking fails."""

    pass


class ModelMonitoringError(MaintainerError):
    """Raised when model monitoring fails."""

    pass


class CodeReviewError(MaintainerError):
    """Raised when code review fails."""

    pass


class DependencyError(MaintainerError):
    """Raised when dependency analysis fails."""

    pass


class TestCoverageError(MaintainerError):
    """Raised when test coverage verification fails."""

    pass
