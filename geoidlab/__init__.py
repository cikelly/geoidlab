'''
GeoidLab - A Python package for geoid modeling and terrain computations.
'''
from pathlib import Path

try:
    try:
        from ._version import __version__
    except ImportError:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
except Exception:
    __version__ = "0.0.0"

__author__ = 'Caleb Kelly'
__email__ = 'geo.calebkelly@gmail.com'

# Read license
with open(Path(__file__).parent.parent / 'LICENSE', 'r') as f:
    __license__ = f.readlines()
    