# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'GeoidLab'
copyright = '2024-2025, Caleb Kelly'
author = 'Caleb Kelly'

# The full version, including alpha/beta/rc tags
try:
    from geoidlab import __version__ as version
    release = version
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath('..'))
    try:
        from geoidlab import __version__ as version
        release = version
    except ImportError:
        version = "0.1.0"
        release = version

# -- General configuration ---------------------------------------------------
# Explicitly set the master document and root doc
master_doc = 'index'
root_doc = 'index'

# Basic project information
project = 'GeoidLab'
copyright = '2024-2025, Caleb Kelly'
author = 'Caleb Kelly'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'myst_parser',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx_copybutton',
]

# Ensure all auto-generated API docs are created
autosummary_generate = True
autodoc_member_order = 'bysource'

# Markdown configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None  # We'll add the logo back once we confirm basic build works
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
