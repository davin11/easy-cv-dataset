# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import inspect
from importlib import import_module
import easy_cv_dataset

project = 'easy-cv-dataset'
copyright = '%Y, Davide Cozzolino'
author = 'Davide Cozzolino'
version = easy_cv_dataset.__version__
release = easy_cv_dataset.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sourcelink = False
html_copy_source = False

linkcode_revision = "main"
linkcode_url = "https://github.com/davin11/easy-cv-dataset/blob/" \
               + linkcode_revision + "/easy_cv_dataset/{filepath}#L{linestart}-L{linestop}"


def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    mod = import_module(info['module'])
    if mod is None:
        return None

    obj = mod
    for part in info['fullname'].split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        filepath = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        filepath = os.path.relpath(filepath, os.path.dirname(easy_cv_dataset.__file__))
    except Exception:
        return None

    
    linestart, linestop = lineno, lineno + len(source) - 1

    return linkcode_url.format(filepath=filepath, linestart=linestart, linestop=linestop)