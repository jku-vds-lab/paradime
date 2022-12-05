"""Custom exceptions for ParaDime.
"""


class NotTrainedError(Exception):
    """Method of DR instance called before training."""


class RelationsNotComputedError(Exception):
    """Attempted to access relation data before computation."""


class NoDatasetRegisteredError(Exception):
    """Attempted to access the dataset before registration."""


class LossNotDeterminedError(Exception):
    """Attempted to access loss before it was determined."""


class UnsupportedConfigurationError(Exception):
    """Combination of parameters and/or objects not supported."""


class SpecificationError(Exception):
    """Specification not supported."""


class CUDANotAvailableError(Exception):
    """CUDA not available."""
