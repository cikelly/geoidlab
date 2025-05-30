'''
GeoidLab - A Python package for geoid modeling and terrain computations.
'''
from pathlib import Path

try:
    from ._version import version as __version__
except ImportError:
    # If the package is not installed in development mode or from source
    __version__ = "0.0.0"

__author__ = 'Caleb Kelly'
__email__ = 'geo.calebkelly@gmail.com'

# Read license
with open(Path(__file__).parent.parent / 'LICENSE', 'r') as f:
    __license__ = f.readlines()
    