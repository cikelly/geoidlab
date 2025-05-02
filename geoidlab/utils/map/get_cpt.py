"""
--------------------------------------------------------------------------------
Name:        get_cpt matplotlib colormap utility
Purpose:     an easy way to fetch .cpt colormap files, based on pycpt

Created:     2020.03
Copyright:   (c) Dimitrios Bouziotas (bouziot)
Licence:     GNU General Public License v3 (GPL-3)
-You may freely copy, distribute and modify the software, in accordance with the provided license.
-You may not sublicense or hold the original author liable. This software comes with no warranty at all.
-Active contributions, forks and redevelopments are welcome.
-If you would like to include this software in your work, please reference it using the zenodo or github link. Please
also reference the original pycpt package (https://github.com/j08lue/pycpt)
--------------------------------------------------------------------------------
"""

__version__ = '0.1.0'
__copyright__ = """(c) 2020 Dimitrios Bouziotas"""
__license__ = "GNU General Public License v3 (GPL-3)"

import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import colorsys

basedir = os.path.join(os.getcwd(), 'cpt')  # base path where cpt files are stored

def get_cmap(cpt_path, name=None, method='cdict', N=None):
    """
    Get the cpt colormap as a LinearSegmented colormap. Utilizes the gmtColormap_openfile parser.

    Parameters
    ----------
    cpt_path : str
        The full directory path to a .cpt file or the filename of a .cpt file in the local repo (check get_cpt.basedir).
    name : str, optional
        Colormap name. If not provided, the name will be derived automatically using _getname().
    method : str, optional
        Choose between 'cdict' and 'list'. 'cdict' fetches all info from the .cpt file. 'list'
        allows you to control the number of colors to sample from.
    N : int, optional
        The number of colors in the colormap to be returned. Only useful when method='list'.

    Returns
    -------
    LinearSegmentedColormap
        The loaded colormap.
    """
    # Derive name if not provided
    if name is None:
        name = _getname(cpt_path)

    # Handle local file path
    if os.path.sep not in cpt_path:  # Case: filename only
        cpt_path = os.path.join(basedir, cpt_path)
    with open(cpt_path) as f:
        return gmtColormap_openfile(f, name=name, method=method, N=N, ret_cmap_type='LinearSegmented')

def get_listed_cmap(cpt_path, name=None, N=None):
    """
    Get the cpt colormap as a ListedColormap. Utilizes the gmtColormap_openfile parser.

    Parameters
    ----------
    cpt_path : str
        The full directory path to a .cpt file or the filename of a .cpt file in the local repo (check get_cpt.basedir).
    name : str, optional
        Colormap name. If not provided, the name will be derived automatically using _getname().
    N : int, optional
        The number of colors in the colormap to be returned. If None, derives colors from the .cpt file.

    Returns
    -------
    ListedColormap
        The loaded colormap.
    """
    # Derive name if not provided
    if name is None:
        name = _getname(cpt_path)

    # Handle local file path
    if os.path.sep not in cpt_path:  # Case: filename only
        cpt_path = os.path.join(basedir, cpt_path)
    with open(cpt_path) as f:
        return gmtColormap_openfile(f, name=name, method='list', N=N, ret_cmap_type='Listed')

def _getname(cpt_path):
    """
    Internal function to fetch the name from a cpt filepath.

    Examples:
    - 'my.mby.cpt' -> 'my_mby'  # Filename
    - 'D:/matplotlib colormaps - cpt-city/cpt/mby.cpt' -> 'mby'  # Full path
    """
    return '_'.join(os.path.basename(cpt_path).split('.')[:-1])

def gmtColormap_openfile(cptf, name=None, method='cdict', N=None, ret_cmap_type='LinearSegmented'):
    """
    Read a GMT color map from an open .cpt file.

    Parameters
    ----------
    cptf : file object
        Open file handle to a .cpt file.
    name : str, optional
        Name for the colormap. If not provided, derived from the file name.
    method : str, optional
        'cdict' or 'list'. 'cdict' generates a LinearSegmentedColormap using a color dictionary.
        'list' generates a colormap using a list of (value, (r,g,b)) tuples.
    N : int, optional
        Number of colors in the colormap. Only used when method='list' or ret_cmap_type='Listed'.
    ret_cmap_type : str, optional
        Type of colormap to return: 'LinearSegmented' or 'Listed'.

    Returns
    -------
    LinearSegmentedColormap or ListedColormap
        The loaded colormap.

    Credits
    -------
    This function originally appears in pycpt, with edits by Dimitrios Bouziotas (2020.03).
    Original work: https://github.com/j08lue/pycpt
    """
    methodnames = ['cdict', 'list']
    ret_cmap_types = ['LinearSegmented', 'Listed']

    # Generate cmap name
    if name is None:
        name = _getname(cptf.name)

    # Process file
    x = []
    r = []
    g = []
    b = []
    lastls = None
    for l in cptf.readlines():
        ls = l.split()

        # Skip empty lines
        if not ls:
            continue

        # Parse header info
        if l[0] in ["#", b"#"]:
            if ls[-1] in ["HSV", b"HSV"]:
                colorModel = "HSV"
                continue
            elif ls[-1] in ["RGB", b"RGB"]:
                colorModel = "RGB"
                continue
            else:  # Skip rogue comments
                continue

        # Skip BFN info
        if ls[0] in ["B", b"B", "F", b"F", "N", b"N"]:
            continue

        # Parse color vectors
        x.append(float(ls[0]))
        r.append(float(ls[1]))
        g.append(float(ls[2]))
        b.append(float(ls[3]))

        # Save last row
        lastls = ls

    # Check if last endrow has the same color; if not, append
    if not ((float(lastls[5]) == r[-1]) and (float(lastls[6]) == g[-1]) and (float(lastls[7]) == b[-1])):
        x.append(float(lastls[4]))
        r.append(float(lastls[5]))
        g.append(float(lastls[6]))
        b.append(float(lastls[7]))

    x = np.array(x)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    if colorModel == "HSV":
        for i in range(r.shape[0]):
            # Convert HSV to RGB
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    elif colorModel == "RGB":
        r /= 255.0
        g /= 255.0
        b /= 255.0

    red = []
    blue = []
    green = []
    xNorm = (x - x[0]) / (x[-1] - x[0])

    # Return colormap
    if method == 'cdict' and ret_cmap_type == 'LinearSegmented':
        for i in range(len(x)):
            red.append([xNorm[i], r[i], r[i]])
            green.append([xNorm[i], g[i], g[i]])
            blue.append([xNorm[i], b[i], b[i]])
        cdict = dict(red=red, green=green, blue=blue)
        return mcolors.LinearSegmentedColormap(name=name, segmentdata=cdict)

    elif method == 'list' and ret_cmap_type == 'LinearSegmented':
        outlist = []
        for i in range(len(x)):
            tup = (xNorm[i], (r[i], g[i], b[i]))
            outlist.append(tup)
        if N and isinstance(N, int):
            return mcolors.LinearSegmentedColormap.from_list(name, outlist, N=N)
        else:
            raise TypeError("Using the method 'list' requires you to set a number of colors N.")

    elif ret_cmap_type == 'Listed':
        pos_out = []
        colors_out = []
        for i in range(len(x)):
            pos_out.append(xNorm[i])
            colors_out.append(mcolors.to_hex((r[i], g[i], b[i])))
        if N and isinstance(N, int) and N <= len(colors_out):
            pos_out = pos_out[:N]
            return pos_out, mcolors.ListedColormap(colors_out, name=name, N=N)
        elif N is None:
            return pos_out, mcolors.ListedColormap(colors_out, name=name)
        else:
            raise TypeError(f"N must be a number of colors less than the actual colors found in the .cpt file ({len(colors_out)} colors found).")

    else:
        raise TypeError(f"method must be one of: {methodnames}, and ret_cmap_type must be one of: {ret_cmap_types}")

def plot_cmaps(cmap_list, width=6, cmap_height=0.5, axes_off=False):
    """
    Plot a colormap or list of colormaps with their names.

    Parameters
    -------
    cmap_list : str, cmap object, or list of cmap objects or strings
        A list of colormaps to plot, either as cmap objects or preinstalled Matplotlib colormap strings.
    width : float
        Width of the plot.
    cmap_height : float
        Height of each colormap in the plot.
    axes_off : bool
        If True, turn off axes.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the colormap display.

    Credits
    -------
    This function originally appears in pycpt, with slight edits by Dimitrios Bouziotas (2020.03).
    Original work: https://github.com/j08lue/pycpt
    """
    if not isinstance(cmap_list, list):
        cmap_list = [cmap_list]

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(nrows=len(cmap_list), figsize=(width, cmap_height * len(cmap_list)))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=0.9)

    if len(cmap_list) == 1:
        cmap = cmap_list[0]
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        axes.imshow(gradient, aspect='auto', cmap=cmap)
        pos = list(axes.get_position().bounds)
        x_text = pos[0] + pos[2] + 0.02
        y_text = pos[1] + pos[3] / 2.0
        fig.text(x_text, y_text, cmap.name, va='center', ha='left', fontsize=12)
        if axes_off:
            axes.set_axis_off()
        return fig

    else:
        for i, cmap in enumerate(cmap_list):
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            axes[i].imshow(gradient, aspect='auto', cmap=cmap)
            pos = list(axes[i].get_position().bounds)
            x_text = pos[0] + pos[2] + 0.02
            y_text = pos[1] + pos[3] / 2.0
            fig.text(x_text, y_text, cmap.name, va='center', ha='left', fontsize=12)
        if axes_off:
            for ax in axes:
                ax.set_axis_off()
        return fig

if __name__ == '__main__':
    # Test 1: Full path, LinearSegmented, method cdict
    cpt_path = r'D:\Users\bouzidi\Desktop\matplotlib colormaps - cpt-city\cpt'
    cpt_fullpath = os.path.join(cpt_path, 'mby.cpt')
    a = get_cmap(cpt_fullpath)
    print(a)

    # Test 2: Local file, change basedir
    basedir = r'D:\Users\bouzidi\Desktop\matplotlib colormaps - cpt-city\test\new_ctp'
    myctp2 = 'purple-orange-d15.cpt'
    pos, b = get_listed_cmap(myctp2)

    # Test 3: Plot
    fig = plot_cmaps([a, b])
    plt.show()