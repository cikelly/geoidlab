'''
GeoidLab - A Python package for geoid modeling and terrain computations.
'''
from importlib.metadata import metadata

from .__version__ import __version__

__author__ = 'Caleb Kelly'
__email__ = 'geo.calebkelly@gmail.com'

# Get license info from package metadata
try:
    __license__ = metadata('geoidlab')['License']
except Exception:
    __license__ = "GPL-3.0-or-later"
    