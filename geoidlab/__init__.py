'''
GeoidLab - A Python package for geoid modeling and terrain computations.
'''
from pathlib import Path

from .__version__ import __version__, __author__, __email__

# Read license
with open(Path(__file__).parent.parent / 'LICENSE', 'r') as f:
    __license__ = f.readlines()
    