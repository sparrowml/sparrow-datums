# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Datums"
_year = dt.datetime.today().strftime("%Y")
copyright = f"{_year} Sparrow Computing LLC"
author = "Sparrow Computing LLC"

# The full version, including alpha/beta/rc tags
release = "0.4.4"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom settings
autosummary_generate = True
add_module_names = False
autodoc_type_aliases = {
    "FloatArray": "FloatArray",
    "PType": "PType",
    "T": "ChunkType",
}
html_theme_options = {
    "sidebar_width": "30%",
    "page_width": "80%",
    "show_powered_by": False,
    "show_relbars": False,
    "show_related": False,
}

numpy_members = set(dir(np.ndarray))


def skip_member(app, what, name, obj, skip, opts):
    """Skip an autosummary member."""
    if name.startswith("__") or name in numpy_members or skip:
        return True
    return None


def setup(app):
    """Set up app."""
    app.connect("autodoc-skip-member", skip_member)
