"""Utility functions for paraDime.

The :mod:`paradime.utils` module implements various utility functions and
classes, such as a mixin for representations, a rporting method, and input
conversion methods.
"""

import functools
import logging
import os
import random
from typing import Any, Optional, Union

import numpy as np
from packaging import version
import torch

from paradime.types import TensorLike

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
)

class _ReprMixin():
    """A mixin implementing a simple __repr__."""

    # based on PyTorch's nn.Module repr
    def __repr__(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            v_str = repr(v)
            v_str = _addindent(v_str, 2)
            lines.append(f"{k}={v_str},")

        main_str = f"{type(self).__name__}("
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

def _addindent(s: str, num_spaces: int) -> str:
    lines = s.split('\n')
    # don't do anything for single-line stuff
    if len(lines) == 1:
        return s
    first = lines.pop(0)
    lines = [(num_spaces * ' ') + line for line in lines]
    s = '\n'.join(lines)
    s = first + '\n' + s
    return s

def log(message: str) -> None:
    """Calls the paradime logger to print a timestamp and a message.
    
    Args:
        message: The message string to print.    
    """
    logger = logging.getLogger('paradime')
    logger.info(message)

def _convert_input_to_numpy(X: Union[TensorLike, list[float]]) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        return X
    elif isinstance(X, list):
        return np.array(X)
    else:
        raise TypeError(f"Input type {type(X)} not supported")

def _convert_input_to_torch(X: Union[TensorLike, list[float]]) -> torch.Tensor:

    if isinstance(X, torch.Tensor):
        return X
    elif isinstance(X, (np.ndarray, list)):
        return torch.tensor(X, dtype=torch.float)
    else:
        raise TypeError(f"Input type {type(X)} not supported")

@functools.cache
def _rowcol_to_triu_index(i: int, j: int, dim: int) -> int:
    if i < j:
        index = round(i * (dim - 1.5) + j - i**2 * 0.5 - 1)
        return index
    elif i > j:
        return _rowcol_to_triu_index(j, i, dim)
    else:
        return -1

@functools.cache
def _get_orig_dim(len_triu: int) -> int:
    return int(np.ceil(np.sqrt(len_triu * 2)))

def seed_all(seed:int) -> torch.Generator:
    """Sets several seeds to maximize reproducibility.

    For infos on reproducibility in PyTorch, see
    https://pytorch.org/docs/stable/notes/randomness.html.
    
    Args:
        seed: The integer to use as a seed.

    Returns:
        The :class:`torch.Generator` instance returned by
        :func:`torch.manual_seed`.
    """
    os.environ['PYTHONHASHSEED']=str(seed)
    if version.parse(torch.version.cuda) >= version.parse("10.2"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return gen
    
def get_color_palette() -> dict[str, str]:
    """Get the custom paraDime color palette.
    
    The palette is usually located in an assets folder in the form of a JSON
    file. If the JSON file is not found, this method attemps to create it from
    parsing an SVG file.

    Returns:
        The color palette as a dict of names and hex color values.

    Raises:
        FileNotFoundError: If neither the JSON nor the SVG file can be found.
    """
    import json
    import os
    import sys

    utils_path = os.path.dirname(__file__)
    assets_path = os.path.abspath(os.path.join(utils_path, '../assets'))
    json_path = os.path.join(assets_path, 'palette.json')
    svg_path = os.path.join(assets_path, 'palette.svg')

    if not os.path.isfile(json_path):
        if os.path.isfile(svg_path):
            sys.path.append(os.path.join(assets_path))
            from make_palette import make_palette # type:ignore
            make_palette(svg_path, json_path)
        else:
            raise FileNotFoundError(
                "Could not find JSON or SVG file to create/import palette."
            )
    with open(json_path, 'r') as f:
            return json.load(f)

def scatterplot(
    coords: TensorLike,
    labels: Optional[TensorLike] = None,
    colormap: Optional[list[str]] = None,
    labels_to_index: Optional[dict] = None,
    figsize: tuple[float, float] = (10,10),
    bgcolor: Optional[str] = "#fcfcfc",
    legend: bool = True,
    legend_options: Optional[dict[str, Any]] = None,
    **kwargs
) -> None:
    """Creates a scatter plot of points at the given coordinates.

    Args:
        coords: The coordinates of the points.
        labels: An list of categorical labels. If labels are given,
            a categorical color scale is used and a legend is constructed
            automatically.
        colormap: A list of colors to use instead of the default categorical
            color scale based on the paraDime palette.
        labels_to_index: A dict that maps labels to indices which are then used
            to access the colors in the categorical color scale.
        figsize: Width and height of the plot in inches.
        bgcolor: The background color of the plot, which by default is also
            to draw thin outlines around the points.
        legend: Whether or not to include the automatically created legend.
        legend_options: A dict of keyword arguments that are passed on to the
            legend method.
        kwargs: Any other keyword arguments are passed on to matplotlib's
            `scatter` method.
    """

    from  matplotlib import pyplot as plt
    from matplotlib import patches
    
    if colormap is None:
        colormap = list(get_color_palette().values())

    colors: Union[str, list[str]]
    if labels is None:
        colors = colormap[0]
    else:
        unique, indices = np.unique(labels, return_inverse=True)
        if labels_to_index is not None:
            indices = [ labels_to_index[i] for i in labels ]
        colors = [ colormap[i] for i in indices ]

        def rect(color: str) -> patches.Rectangle:
            return patches.Rectangle((0.,0.), 1., 1., fc=color)

        rects = [ rect(colormap[i]) for i in np.unique(indices) ]

    fig = plt.figure(figsize=figsize)
    dpi: float = fig.dpi

    if hasattr(kwargs, 's'):
        pointsize = kwargs['s']
    else:
        min_figsize = min(figsize)
        pointwidth = min(
            max(
                0.005 * min_figsize * dpi,
                min_figsize * dpi / np.sqrt(len(coords)) * 0.2
            ),
            0.02 * min_figsize * dpi
        )
        pointsize = pointwidth**2
    
    ax = fig.add_subplot(111)
    ax.set_facecolor(bgcolor)

    scatter_kwargs = {
        'c': colors,
        's': pointsize,
        'edgecolor': bgcolor,
    }
    if kwargs:
        scatter_kwargs = {**scatter_kwargs, **kwargs}

    points = ax.scatter(
        x=coords[:,0],
        y=coords[:,1],
        **scatter_kwargs
    )
    ax.set_aspect('equal')
    ax.axis('off');
    points.set_linewidths(.08 * np.sqrt(pointsize))

    if legend and labels is not None:
        legend_kwargs = {
            'handleheight': 1.,
            'handlelength': 1.,
            'loc': 4,
            'fancybox': False,
        }
        if legend_options is not None:
            legend_kwargs = {**legend_kwargs, **legend_options}
        
        ax.legend(rects, unique, **legend_kwargs)

    plt.show()

    return fig
    