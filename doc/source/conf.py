# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
project = 'PythonSI'
copyright = '2025, PSI Contributors'
author = 'PSI Contributors'
release = 'First pre-release'

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../psi'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage',
              'sphinx.ext.napoleon', 'sphinx.ext.autosummary']

autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
