"""Typed hard failures used by pipeline acceptance gates."""


class PipelineError(RuntimeError):
    """Base class for a pipeline-stopping validation failure."""


class ConfigurationError(PipelineError):
    """Configuration is incomplete, inconsistent, or unconfirmed."""


class SchemaError(PipelineError):
    """A standardized table violates its contract."""


class LeakageError(PipelineError):
    """A forbidden or validation-derived value can enter model fitting."""


class LinkageError(PipelineError):
    """An event cannot be linked safely and uniquely."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


class IntegrityError(PipelineError):
    """Rows, folds, features, probabilities, or hashes are inconsistent."""


class CountMismatchError(IntegrityError):
    """Paper counts failed after diagnostic evidence was constructed."""

    def __init__(self, message, *, attrition, comparison):
        super().__init__(message)
        self.attrition = attrition
        self.comparison = comparison


class UnitError(PipelineError):
    """Measurement units are incompatible with confirmed rules."""
