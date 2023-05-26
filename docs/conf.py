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
from smarts import VERSION

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "SMARTS"
copyright = "2021, Huawei Technologies."
author = "Huawei Noah's Ark Lab."

# The full version, including alpha/beta/rc tags
release = VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # support Markdown-based documentation
    "sphinx.ext.autodoc",  # automatically extract docs from docstrings
    "sphinx.ext.coverage",  # to generate documentation coverage reports
    "sphinx.ext.extlinks",  # shorten external links
    "sphinx.ext.napoleon",  # support Numpy and Google doc style
    "sphinx.ext.todo",  # support for todo items
    "sphinx.ext.viewcode",  # link to sourcecode from docs
    "sphinx_rtd_theme",  # Read The Docs theme
    "sphinx_click",  # extract documentation from a `click` application
    "sphinxcontrib.apidoc",
    "sphinxcontrib.spelling",
]

extlinks = {
    "examples": (
        "https://github.com/huawei-noah/SMARTS/blob/master/examples/%s",
        "%s",
    ),
    "scenarios": (
        "https://github.com/huawei-noah/SMARTS/blob/master/scenarios/%s",
        "%s",
    ),
}

# Configuring automated generation of api documentation.
# See: https://github.com/sphinx-contrib/apidoc
apidoc_module_dir = ".."
apidoc_module_first = True
apidoc_excluded_paths = [
    "cli",
    "examples",
    "setup.py",
    "scenarios",
    "smarts/ros",
    "zoo/policies/interaction_aware_motion_prediction",
]
apidoc_extra_args = [
    "--force",
    "--separate",
    "--ext-viewcode",
    "--doc-project=SMARTS",
    "--maxdepth=2",
    "--templatedir=_templates/apidoc",
]
autodoc_mock_imports = [
    "av2",
    "cpuinfo",
    "cv2",
    "gymnasium",
    "lxml",
    "mdutils",
    "moviepy",
    "opendrive2lanelet",
    "pandas",
    "pathos",
    "PIL",
    "pynput",
    "ray",
    "rich",
    "tabulate",
    "tools",
    "torch",
]

todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for nit-pick (-n) ----------------------------------------------------

nitpick_ignore = {
    ("py:class", "optional"),
    ("py:class", "ellipsis"),
    ("py:class", "function"),
    # Most of these dynamic type ignores would go away in python>=3.10
    # See for more context: https:github.com/sphinx-doc/sphinx/issues/10090
    ("py:class", "T"),
    ("py:class", "Score"),
    ("py:class", "Done"),
    ("py:class", "CostFuncs"),
    ("py:class", "ActType"),
    ("py:class", "ObsType"),
    ("py:class", "smarts.env.gymnasium.wrappers.metric.utils.T"),
}
nitpick_ignore_regex = {
    (r"py:class", r"av2\..*"),
    (r"py:class", r"google\.protobuf\..*"),
    (r"py:class", r"grpc\..*"),
    (r"py:class", r"gym\..*"),
    (r"py:class", r"gymnasium\..*"),
    (r"py:class", r"logging\..*"),
    (r"py:class", r"multiprocessing\..*"),
    (r"py:class", r"np\..*"),
    (r"py:class", r"numpy\..*"),
    (r"py:class", r"opendrive2lanelet\..*"),
    (r"py:class", r"panda3d\..*"),
    (r"py.class", r"pathlib\..*"),
    (r"py.class", r"pybullet(_utils)?\..*"),
    (r"py.class", r"re\..*"),
    (r"py:class", r"shapely\..*"),
    (r"py:class", r"sumo(lib)?\..*"),
    (r"py:class", r"tornado\..*"),
    (r"py:class", r"traci\..*"),
    (r"py:exc", r"configparser\..*"),
}

# -- Options for spelling ----------------------------------------------------
spelling_exclude_patterns = ["ignored_*", "**/*_pb2*"]
spelling_ignore_pypi_package_names = True
spelling_show_suggestions = True
spelling_suggestion_limit = 2
spelling_ignore_contributor_names = False
spelling_word_list_filename = ["spelling_wordlist.txt"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
