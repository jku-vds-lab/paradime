"""Custom exceptions for paraDime.
"""

class NotTrainedError(Exception):
    """Method of DR instance called before training."""

class RelationsNotComputedError(Exception):
    """Attempted to access relation data before computation."""

class NoDatasetRegisteredError(Exception):
    """Attempted to access the dataset before registration."""

class UnsupportedConfigurationError(Exception):
    """Combination of parameters and/or objects not supported."""