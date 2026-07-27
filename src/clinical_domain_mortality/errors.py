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


class IntegrityError(PipelineError):
    """Rows, folds, features, probabilities, or hashes are inconsistent."""


class UnitError(PipelineError):
    """Measurement units are incompatible with confirmed rules."""
