import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Leadership Emergence ABM'
copyright = '2024, Leadership Emergence ABM Project Contributors'
author = 'Leadership Emergence ABM Project Contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
