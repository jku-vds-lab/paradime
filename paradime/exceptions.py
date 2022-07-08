class NotTrainedError(Exception):
    """Method of DR instance called before training."""

class RelationsNotComputedError(Exception):
    """Method of DR instance called before computing relations."""

class NoDatasetRegisteredError(Exception):
    """Method of DR instance called before registering a dataset."""