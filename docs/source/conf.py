# type: ignore

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("sphinxext"))

from ghlink import make_linkcode_resolve


# -- Project information -----------------------------------------------------

project = "paradime"
copyright = "2022, Andreas Hinterreiter"
author = "Andreas Hinterreiter"


def get_version():
    for line in open("../../paradime/_version.py").readlines():
        mo = re.match(r"^__version__ = [\"']([^\"']*)[\"']", line)
        if mo:
            return mo.group(1)


release = get_version()
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "sphinx-favicon",
    "nbsphinx",
    "nbsphinx_link",
]

nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "../../assets/ParaDime.svg"
html_theme_options = {
    "logo_only": True,
}

# Favicons (as per https://github.com/tcmetzger/sphinx-favicon)

favicons = [
    {
        "rel": "icon",
        "static-file": "favicons/favicon.svg",
        "type": "image/svg+xml",
    },
    {
        "rel": "icon",
        "sizes": "16x16",
        "static-file": "favicons/favicon16.png",
        "type": "image/png",
    },
    {
        "rel": "icon",
        "sizes": "32x32",
        "static-file": "favicons/favicon32.png",
        "type": "image/png",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "static-file": "favicons/favicon180.png",
        "type": "image/png",
    },
]

# Mappings for intersphinx
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "paradime",
    "https://github.com/einbandi/paradime/blob/"
    "{revision}/{package}/{path}#L{lineno}",
)
