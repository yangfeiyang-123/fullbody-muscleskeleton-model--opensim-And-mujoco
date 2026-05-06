class DependencyMissingError(RuntimeError):
    """Raised when an optional biomechanics backend is not installed."""


class ModelLoadError(RuntimeError):
    """Raised when a model file cannot be loaded."""


class CheckSkipped(RuntimeError):
    """Raised by a check when required inputs are unavailable."""
