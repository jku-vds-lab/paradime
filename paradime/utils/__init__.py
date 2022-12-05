"""Utility functions for ParaDime.

The :mod:`paradime.utils` subpackage includes various modules that implement
utility functions for logging, plotting, and input conversion.
"""

from paradime.utils import convert
from paradime.utils import logging
from paradime.utils import parsing
from paradime.utils import plotting
from paradime.utils import seed


class _ReprMixin:
    """A mixin implementing a simple __repr__."""

    def __repr__(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            v_str = repr(v)
            v_str = _addindent(v_str, 2)
            lines.append(f"{k}={v_str},")

        main_str = f"{type(self).__name__}("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


def _addindent(s: str, num_spaces: int) -> str:
    lines = s.split("\n")
    # don't do anything for single-line stuff
    if len(lines) == 1:
        return s
    first = lines.pop(0)
    lines = [(num_spaces * " ") + line for line in lines]
    s = "\n".join(lines)
    s = first + "\n" + s
    return s
