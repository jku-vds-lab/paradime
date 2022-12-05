"""Random seeding for ParaDime.

The :mod:`paradime.utils.seed` subpackage implements a function to seed all
random number generators potentially involved in a ParaDime routine.
"""

import os
import random

import numpy as np
from packaging import version
import torch


def seed_all(seed: int) -> torch.Generator:
    """Sets several seeds to maximize reproducibility.

    For infos on reproducibility in PyTorch, see
    https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed: The integer to use as a seed.

    Returns:
        The :class:`torch.Generator` instance returned by
        :func:`torch.manual_seed`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        if version.parse(torch.version.cuda) >= version.parse("10.2"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return gen
