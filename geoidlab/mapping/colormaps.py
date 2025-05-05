############################################################
# Utilities for custom colormaps                           #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pathlib import Path


def parula_cmap() -> LinearSegmentedColormap:
    '''
    Create MATLAB's parula colormap
    
    Parameters
    ----------
    reverse   : bool
                If True, reverse the colormap.
                
    parula_cmap colormap is based on MATLAB's 'parula' colormap.
    MATLAB (R) is a registered trademark of The MathWorks, Inc.
    See: https://www.mathworks.com/help/matlab/ref/colormap.html
    '''
    parula_colors = np.array([
    [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270], [0.2081, 0.2386, 0.6771],
    [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792], [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683],
    [0.0117, 0.3875, 0.8820], [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467], [0.0779, 0.5040, 0.8384],
    [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8263], [0.0641, 0.5570, 0.8240], [0.0488, 0.5772, 0.8228],
    [0.0343, 0.5966, 0.8199], [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436], [0.0590, 0.6838, 0.7254],
    [0.0843, 0.6928, 0.7062], [0.1133, 0.7015, 0.6859], [0.1453, 0.7098, 0.6646], [0.1801, 0.7177, 0.6424],
    [0.2178, 0.7250, 0.6193], [0.2586, 0.7317, 0.5954], [0.3022, 0.7376, 0.5712], [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244], [0.4420, 0.7481, 0.5033], [0.4871, 0.7491, 0.4840], [0.5300, 0.7491, 0.4665],
    [0.5709, 0.7485, 0.4494], [0.6099, 0.7473, 0.4327], [0.6473, 0.7456, 0.4164], [0.6834, 0.7435, 0.4003],
    [0.7184, 0.7411, 0.3844], [0.7525, 0.7384, 0.3686], [0.7858, 0.7356, 0.3529], [0.8185, 0.7327, 0.3374],
    [0.8507, 0.7299, 0.3219], [0.8824, 0.7274, 0.3064], [0.9139, 0.7258, 0.2906], [0.9450, 0.7261, 0.2743],
    [0.9739, 0.7314, 0.2574], [0.9938, 0.7455, 0.2397], [0.9990, 0.7653, 0.2224], [0.9955, 0.7861, 0.2060],
    [0.9880, 0.8066, 0.1908], [0.9789, 0.8271, 0.1768], [0.9697, 0.8481, 0.1633], [0.9626, 0.8705, 0.1498],
    [0.9589, 0.8949, 0.1353], [0.9598, 0.9218, 0.1208], [0.9661, 0.9514, 0.1050], [0.9763, 0.9831, 0.0856]
    ])
    
    # if reverse:
    #     return LinearSegmentedColormap.from_list('parula', np.flipud(parula_colors))
    
    return LinearSegmentedColormap.from_list('parula', parula_colors)

def bright_rainbow_cmap(n=256) -> ListedColormap:
    '''
    Create a bright rainbow colormap
    
    Returns
    -------
    cmap      : LinearSegmentedColormap
                Bright rainbow colormap
    '''
    colors = np.array([
        [1.0, 0.0, 0.0],   # Red
        [1.0, 0.5, 0.0],   # Orange
        [1.0, 1.0, 0.0],   # Yellow
        [0.0, 1.0, 0.0],   # Green
        [0.0, 1.0, 1.0],   # Cyan
        [0.0, 0.0, 1.0],   # Blue
        [1.0, 0.0, 1.0]    # Magenta
    ])
    
    # Interpolation
    xp = np.linspace(0, 1, colors.shape[0])  # original stops
    x = np.linspace(0, 1, n)  # desired resolution
    R = np.interp(x, xp, colors[:, 0])
    G = np.interp(x, xp, colors[:, 1])
    B = np.interp(x, xp, colors[:, 2])

    cmap_array = np.vstack([R, G, B]).T
    cmap = ListedColormap(cmap_array)
    
    return cmap

def cpt_cmap(cpt_name: str = 'GMT_rainbow') -> LinearSegmentedColormap:
    '''
    Load a GMT .cpt file as a Matplotlib colormap
    
    Parameters
    ----------
    name      : Name of .cpt file
    
    Returns
    -------
    cmap      : LinearSegmentedColormap
                Matplotlib colormap
    '''
    from .get_cpt import get_cmap
    
    CPT_BASE_DIR = Path(__file__).resolve().parent / 'cpt'
    VALID_CPT_FILES = [f.stem for f in CPT_BASE_DIR.iterdir() if f.suffix == '.cpt']
    
    cpt_name = Path(cpt_name)
    if cpt_name.suffix != '.cpt':
        cpt_name = cpt_name.with_suffix('.cpt')

    cpt_path = CPT_BASE_DIR / cpt_name
    
    if not cpt_path.exists():
        raise FileNotFoundError(f'Could not find {cpt_name} in {CPT_BASE_DIR}. \nValid cpt: {VALID_CPT_FILES}')
    
    return get_cmap(str(cpt_path))