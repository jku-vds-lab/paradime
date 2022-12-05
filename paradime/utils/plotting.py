"""Plotting utilities for ParaDime.

The :mod:`paradime.utils.plotting` module implements plotting functions and
color palette retrieval.
"""

from typing import Any, Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib.colors
    import matplotlib.axes

import numpy as np

from paradime.types import TensorLike
from paradime.utils.make_palette import make_palette

def get_color_palette() -> dict[str, str]:
    """Get the custom ParaDime color palette.
    
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

    utils_path = os.path.dirname(__file__)
    json_path = os.path.join(utils_path, 'palette.json')
    svg_path = os.path.join(utils_path, 'palette.svg')

    if not os.path.isfile(json_path):
        if os.path.isfile(svg_path):
            make_palette(svg_path, json_path)
        else:
            raise FileNotFoundError(
                "Could not find JSON or SVG file to create/import palette."
            )
    with open(json_path, 'r') as f:
            return json.load(f)

def get_colormap() -> "matplotlib.colors.ListedColormap":
    from ._cmap import _paradime_cmap
    return _paradime_cmap

def scatterplot(
    coords: TensorLike,
    labels: Optional[TensorLike] = None,
    colormap: Optional[list[str]] = None,
    labels_to_index: Optional[dict] = None,
    figsize: tuple[float, float] = (10,10),
    bgcolor: Optional[str] = "#fcfcfc",
    legend: bool = True,
    legend_options: Optional[dict[str, Any]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs
) -> None:
    """Creates a scatter plot of points at the given coordinates.

    Args:
        coords: The coordinates of the points.
        labels: An list of categorical labels. If labels are given,
            a categorical color scale is used and a legend is constructed
            automatically.
        colormap: A list of colors to use instead of the default categorical
            color scale based on the ParaDime palette.
        labels_to_index: A dict that maps labels to indices which are then used
            to access the colors in the categorical color scale.
        figsize: Width and height of the plot in inches.
        bgcolor: The background color of the plot, which by default is also
            to draw thin outlines around the points.
        legend: Whether or not to include the automatically created legend.
        legend_options: A dict of keyword arguments that are passed on to the
            legend method.
        ax: An axes of the current figure. This argument is useful if the
            scatterplot should be added to an existing figure.
        kwargs: Any other keyword arguments are passed on to matplotlib's
            `scatter` method.

    Returns:
        The :class:`matplotlib.axes.Axes` instance of the plot.
    """

    from matplotlib import pyplot as plt
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

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    else:
        assert isinstance(ax, plt.Axes)
        fig = ax.get_figure()

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
    ax.set_axis_off()
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

    return ax
    