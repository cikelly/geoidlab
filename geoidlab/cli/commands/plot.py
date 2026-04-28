import argparse
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from typing import Callable
from pathlib import Path
from numbers import Number
from matplotlib.colors import Colormap, LightSource, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.transforms import Bbox

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CARTOPY = False

from geoidlab.mapping.colormaps import parula_cmap, bright_rainbow_cmap, cpt_cmap
from geoidlab.cli.commands.utils.common import directory_setup

CUSTOM_CMAPS = {
    'parula': parula_cmap(),
    'bright_rainbow': bright_rainbow_cmap()
}

# Unit conversion from meters
UNIT_CONVERSIONS = {
    'cm': 100,
    'mm': 1000
}

cpt_list = cpt_cmap(cpt_list=True)


def load_boundary_data(boundary_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Load boundary linework from a CSV or shapefile.

    Returns
    -------
    segments : list of (lon, lat) arrays
    '''
    path = Path(boundary_path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        boundary_data = pd.read_csv(path)
        if 'lon' not in boundary_data.columns or 'lat' not in boundary_data.columns:
            raise ValueError(f'Boundary file {boundary_path} must contain "lon" and "lat" columns.')
        return [(boundary_data['lon'].to_numpy(), boundary_data['lat'].to_numpy())]

    if suffix == '.shp':
        try:
            import shapefile
        except ImportError as exc:
            raise ImportError(
                'Shapefile boundary plotting requires the "pyshp" package.'
            ) from exc

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        reader = shapefile.Reader(str(path))
        for shape in reader.shapes():
            if not shape.points:
                continue
            points = np.asarray(shape.points, dtype=float)
            parts = list(shape.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                segment = points[start:end]
                if len(segment) == 0:
                    continue
                segments.append((segment[:, 0], segment[:, 1]))

        if not segments:
            raise ValueError(f'No plottable linework found in shapefile {boundary_path}.')
        return segments

    raise ValueError(
        f'Unsupported boundary file format: {boundary_path}. Supported formats are .csv and .shp.'
    )


def build_relief_image(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    cmap: Colormap,
    vmin: float | None,
    vmax: float | None,
    azdeg: float,
    altdeg: float,
    exaggeration: float,
) -> tuple[np.ndarray, Normalize, str]:
    '''
    Build a shaded-relief RGB image from gridded data.
    '''
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    data = np.asarray(data, dtype=float)

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError('Cannot build relief image from an array with no finite values.')

    data_min = float(np.nanmin(finite)) if vmin is None else float(vmin)
    data_max = float(np.nanmax(finite)) if vmax is None else float(vmax)
    if np.isclose(data_min, data_max):
        data_max = data_min + 1e-12

    norm = Normalize(vmin=data_min, vmax=data_max)
    shaded_data = np.nan_to_num(data, nan=data_min)

    dx = float(np.nanmedian(np.abs(np.diff(lon)))) if lon.size > 1 else 1.0
    dy = float(np.nanmedian(np.abs(np.diff(lat)))) if lat.size > 1 else 1.0
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    light_source = LightSource(azdeg=azdeg, altdeg=altdeg)
    relief_image = light_source.shade(
        shaded_data,
        cmap=cmap,
        norm=norm,
        blend_mode='overlay',
        vert_exag=exaggeration,
        dx=dx,
        dy=dy,
    )

    if np.isnan(data).any():
        relief_image = relief_image.copy()
        relief_image[np.isnan(data), -1] = 0.0

    origin = 'lower' if lat[0] <= lat[-1] else 'upper'
    return relief_image, norm, origin


def build_surface_coordinates(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    exaggeration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Convert lon/lat/elevation grids to locally scaled surface coordinates in km.

    The vertical axis is visually scaled relative to the map span so oblique
    surface plots read like a terrain block diagram instead of a tilted map.
    '''
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    data = np.asarray(data, dtype=float)

    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))
    km_per_deg_lat = 111.32
    km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(lat0))
    if np.isclose(km_per_deg_lon, 0.0):
        km_per_deg_lon = km_per_deg_lat

    x = (lon - lon0) * km_per_deg_lon
    y = (lat - lat0) * km_per_deg_lat
    x2d, y2d = np.meshgrid(x, y)

    finite = data[np.isfinite(data)]
    z_fill = float(np.nanmin(finite)) if finite.size else 0.0
    z_km = np.nan_to_num(data, nan=z_fill) / 1000.0

    z_min = float(np.nanmin(z_km))
    z_max = float(np.nanmax(z_km))
    z_range = max(z_max - z_min, 1e-6)
    horizontal_span = max(np.ptp(x), np.ptp(y), 1.0)

    # Scale the relief to a visible fraction of the horizontal map span.
    vertical_span = 0.25 * horizontal_span * exaggeration
    z2d = ((z_km - z_min) / z_range) * vertical_span

    return x2d, y2d, z2d, z_min


def parse_save_pad(values) -> tuple[float | None, float | None, float | None, float | None]:
    '''
    Parse save padding as LEFT [RIGHT] [BOTTOM] [TOP].
    Values may be numeric or the string "None".
    '''
    if values is None:
        return (None, None, None, None)

    parsed = []
    for value in values[:4]:
        if isinstance(value, str) and value.lower() == 'none':
            parsed.append(None)
        else:
            parsed.append(float(value))

    while len(parsed) < 4:
        parsed.append(None)

    return tuple(parsed)


def apply_save_pad(fig, renderer, bbox, save_pad) -> Bbox:
    '''
    Apply asymmetric per-side padding/cropping to a bbox in inches.
    Positive values expand outward; negative values crop inward.
    '''
    if save_pad is None:
        return bbox

    left, right, bottom, top = save_pad
    x0, y0, x1, y1 = bbox.extents

    if isinstance(left, Number):
        x0 -= left
    if isinstance(right, Number):
        x1 += right
    if isinstance(bottom, Number):
        y0 -= bottom
    if isinstance(top, Number):
        y1 += top

    return Bbox.from_extents(x0, y0, x1, y1)


def add_panel_colorbar(fig, ax, mappable, label: str, args, use_surface: bool=False):
    '''
    Add a per-panel colorbar with side or inset-corner placement.
    '''
    inset_locations = {'upper-right', 'lower-right', 'upper-left', 'lower-left'}

    if args.cbar_location in inset_locations:
        shrink = max(args.cbar_shrink, 0.05)
        pad = args.cbar_pad if args.cbar_pad is not None else 0.01
        edge_margin = 0.01
        # 3D axes can mutate their active drawing box as the camera and aspect
        # change. Use the original subplot slot so corner colorbars stay
        # anchored to the requested corner instead of drifting across the figure.
        ax_pos = ax.get_position(original=True).frozen()
        ax_x0, ax_y0, ax_w, ax_h = ax_pos.bounds

        if args.cbar_orientation == 'vertical':
            cbar_w = 0.018
            cbar_h = ax_h * max(0.18, min(0.9, 0.5 * shrink))
            if 'right' in args.cbar_location:
                x0 = ax_x0 + ax_w - cbar_w - pad
            else:
                x0 = ax_x0 + pad

            if 'upper' in args.cbar_location:
                y0 = ax_y0 + ax_h - cbar_h - edge_margin
            else:
                y0 = ax_y0 + edge_margin
        else:
            cbar_w = ax_w * max(0.18, min(0.9, 0.5 * shrink))
            cbar_h = 0.018
            if 'right' in args.cbar_location:
                x0 = ax_x0 + ax_w - cbar_w - edge_margin
            else:
                x0 = ax_x0 + edge_margin

            if 'upper' in args.cbar_location:
                y0 = ax_y0 + ax_h - cbar_h - pad
            else:
                y0 = ax_y0 + pad

        cax = fig.add_axes([x0, y0, cbar_w, cbar_h])
        cax.set_in_layout(False)
        cbar = fig.colorbar(mappable, cax=cax, orientation=args.cbar_orientation)
        cbar.ax.set_in_layout(False)
    else:
        colorbar_kwargs = {
            'orientation': args.cbar_orientation,
            'shrink': args.cbar_shrink,
        }
        if args.cbar_pad is not None:
            colorbar_kwargs['pad'] = args.cbar_pad
        elif use_surface:
            colorbar_kwargs['pad'] = 0.02 if args.cbar_orientation == 'horizontal' else 0.03
        if args.cbar_location in {'left', 'right', 'top', 'bottom'}:
            colorbar_kwargs['location'] = args.cbar_location
        cbar = fig.colorbar(mappable, ax=ax, **colorbar_kwargs)

    cbar.set_label(label)
    return cbar


def get_colormap(cmap_name: str) -> Colormap:
    '''Retrieve colormap by name, handling custom and GMT .cpt colormaps'''
    if cmap_name in CUSTOM_CMAPS:
        return CUSTOM_CMAPS[cmap_name]
    elif cmap_name.endswith('.cpt'):
        return cpt_cmap(cmap_name)
    else:
        try:
            return plt.get_cmap(cmap_name)
        except ValueError:
            raise ValueError(f'Invalid colormap: {cmap_name}. Use --list-cmaps to see available options.')

def list_colormaps() -> None:
    '''Print available colormaps.'''
    print('Available colormaps:')
    print('- Standard Matplotlib colormaps (e.g., viridis, plasma, etc.)')
    print('- Custom colormaps:', ', '.join(CUSTOM_CMAPS.keys()))
    print('- GMT .cpt colormaps:', ', '.join(cpt_list))
    
def nice_scale_length(range_size: float) -> float:
    '''Return a nice scalebar length (e.g., 10, 50, 100) based on range size'''
    candidates = [1, 2, 5, 10, 20, 50, 100, 200, 250]
    target = range_size * 0.2  # 10% of the range
    return min(candidates, key=lambda x: abs(x - target))

def add_north_arrow(ax, x=0.95, y=0.95, size=30, color='black') -> None:
    ax.annotate('N', xy=(x, y), xytext=(0, size),
               arrowprops=dict(arrowstyle='->', color=color),
               xycoords=ax.transAxes, textcoords='offset points',
               ha='center', va='center', fontsize=size//2, 
               fontweight='bold', color=color)
    
def add_plot_arguments(parser) -> None:
    '''Add plotting arguments to an ArgumentParser instance'''
    parser.add_argument('-f', '--filename', type=str, nargs='+', help='NetCDF file(s) to plot. Can specify multiple files for subplot layout.')
    parser.add_argument('-v', '--variable', action='append', type=str, help='Variable name(s) to plot')
    parser.add_argument('-c', '--cmap', type=str, nargs='+', help='Colormap(s) to use. Can specify multiple colormaps for multiple files. For GMT .cpt files, use the file name with extension.', default=['GMT_rainbow.cpt'])
    parser.add_argument('--fig-size', type=float, nargs=2, default=[5, 5], help='Figure size in inches')
    parser.add_argument('--vmin', type=float, help='Minimum value for colorbar')
    parser.add_argument('--vmax', type=float, help='Maximum value for colorbar')
    parser.add_argument('--font-size', type=int, default=10, help='Font size for labels')
    parser.add_argument('--title', type=str, nargs='+', default=None, help='Title(s) for the figure. Can specify multiple titles for multiple files.')
    parser.add_argument('--title-font-size', type=int, default=12, help='Font size for title')
    parser.add_argument('--font-family', type=str, default='Arial', help='Font family for labels')
    parser.add_argument('--cbar-title', type=str, default=None, help='Title for colorbar')
    parser.add_argument('--cbar-orientation', type=str, default='vertical', choices=['horizontal', 'vertical'], help='Orientation for per-panel colorbars.')
    parser.add_argument('--cbar-shrink', type=float, default=1.0, help='Shrink factor for per-panel colorbars.')
    parser.add_argument('--cbar-pad', type=float, default=None, help='Padding between the plot and per-panel colorbar.')
    parser.add_argument(
        '--cbar-location',
        type=str,
        default='right',
        choices=['left', 'right', 'top', 'bottom', 'upper-right', 'lower-right', 'upper-left', 'lower-left'],
        help='Location for per-panel colorbars. Side locations place the colorbar outside the axes; corner locations place it inside the plot.'
    )
    parser.add_argument('--list-cmaps', action='store_true', help='List available colormaps and exit')
    parser.add_argument('--save', action='store_true', help='Save figure')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saving figure')
    parser.add_argument(
        '--save-pad',
        nargs='*',
        default=None,
        help='Extra save padding/cropping in inches as LEFT [RIGHT] [BOTTOM] [TOP]. '
             'Use numeric values or "None". Positive expands outward; negative crops inward. '
             'Examples: --save-pad 0.2, --save-pad 0.1 -0.1, --save-pad None -0.2 None None'
    )
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject', help='Name of the project')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help='X-axis limits')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='Y-axis limits')
    parser.add_argument('--scalebar', action='store_true', help='Show scalebar')
    parser.add_argument('--scalebar-units', type=str, default='km', choices=['km', 'degrees'], help='Scalebar units')
    parser.add_argument('--scalebar-fancy', action='store_true', help='Use fancy scalebar')
    parser.add_argument('-u', '--unit', type=str, default=None, choices=['m', 'cm', 'mm'], help='Unit to display data with length units')
    parser.add_argument('--relief', action='store_true', help='Render the grid as a shaded-relief 2D map instead of pcolormesh.')
    parser.add_argument('--relief-exaggeration', type=float, default=10.0, help='Vertical exaggeration used for shaded relief.')
    parser.add_argument('--relief-azdeg', type=float, default=135.0, help='Illumination azimuth in degrees for shaded relief.')
    parser.add_argument('--relief-altdeg', type=float, default=45.0, help='Illumination altitude in degrees for shaded relief.')
    parser.add_argument('--surface', action='store_true', help='Render the grid as an oblique surface on 3D axes.')
    parser.add_argument('--surface-exaggeration', type=float, default=0.5, help='Vertical exaggeration used for surface rendering.')
    parser.add_argument('--surface-elev', type=float, default=30.0, help='Camera elevation angle in degrees for surface rendering.')
    parser.add_argument('--surface-azim', type=float, default=-110.0, help='Camera azimuth angle in degrees for surface rendering.')
    parser.add_argument('--boundary', type=str, help='Boundary file to overlay (.csv with lon/lat columns or .shp shapefile)')
    parser.add_argument('--bound-color', type=str, default='k', help='Color for boundary lines (default: k)')
    parser.add_argument('--bound-linewidth', type=float, default=1.2, help='Line width for boundary lines (default: 1.2)')
    parser.add_argument('--sharex', action='store_true', help='Share x-axis between subplots')
    parser.add_argument('--sharey', action='store_true', help='Share y-axis between subplots')
    parser.add_argument('--nrows', type=int, default=None, help='Number of subplot rows. If not provided, rows are determined automatically.')
    parser.add_argument('--ncols', type=int, default=None, help='Number of subplot columns. If not provided, columns are determined automatically.')
    parser.add_argument('--global-plot', action='store_true', help='Enable global map styling with coastlines (requires Cartopy).')
    parser.add_argument('--projection', type=str, default='PlateCarree', choices=['PlateCarree', 'Robinson', 'Mollweide'], help='Projection to use when global plot is enabled (requires Cartopy).')
    parser.add_argument('--global-cbar-orientation', type=str, default='horizontal', choices=['horizontal', 'vertical'], help='Colorbar orientation to use when global plot is enabled.')
    parser.add_argument('--global-cbar-shrink', type=float, default=0.6, help='Shrink factor for colorbar when global plot is enabled.')
    parser.add_argument('--global-cbar-pad', type=float, default=0.05, help='Padding between axes and colorbar when global plot is enabled.')
    parser.add_argument('--share-cbar', action='store_true', help='Use one shared colorbar for all visible subplots instead of one colorbar per subplot.')
    parser.add_argument('--shared-cbar-orientation', type=str, default='vertical', choices=['horizontal', 'vertical'], help='Orientation for a shared colorbar.')
    parser.add_argument('--shared-cbar-shrink', type=float, default=0.5, help='Shrink factor for a shared colorbar.')
    parser.add_argument('--shared-cbar-pad', type=float, default=0.02, help='Padding between subplot group and shared colorbar.')
    parser.add_argument('--shared-cbar-font-size', type=int, default=12, help='Font size for shared colorbar label and tick labels.')
    parser.add_argument('--contour', action='store_true', help='Overlay contours on the plotted field.')
    parser.add_argument('--contour-color', type=str, default='black', help='Contour line color.')
    parser.add_argument('--contour-linewidth', type=float, default=0.25, help='Contour line width.')
    parser.add_argument('--contour-alpha', type=float, default=0.8, help='Contour transparency.')
    parser.add_argument('--contour-levels', type=str, default=None, help='Contour levels as an integer count or a comma-separated list of explicit values.')

def parse_contour_levels(levels: str | None):
    '''Parse contour levels from CLI/config input.'''
    if levels is None:
        return None

    levels = str(levels).strip()
    if not levels:
        return None

    if ',' in levels:
        parsed_levels = []
        for level in levels.split(','):
            level = level.strip()
            if not level:
                continue
            parsed_levels.append(float(level))
        if not parsed_levels:
            raise ValueError('Contour levels list is empty.')
        return parsed_levels

    try:
        return int(levels)
    except ValueError:
        return [float(levels)]

def main(args=None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description='Plot a NetCDF file')
        add_plot_arguments(parser)
        args = parser.parse_args()
        
    if getattr(args, 'list_cmaps', False):
        list_colormaps()
        return 0

    contour_levels = parse_contour_levels(getattr(args, 'contour_levels', None))
    try:
        save_pad = parse_save_pad(args.save_pad)
    except ValueError as exc:
        print(f'Warning: Invalid --save-pad value ({exc}). Ignoring save padding override.')
        save_pad = (None, None, None, None)
    if args.cbar_shrink <= 0:
        print('Warning: --cbar-shrink must be positive. Using default value of 1.0.')
        args.cbar_shrink = 1.0
    if args.relief and args.relief_exaggeration <= 0:
        print('Warning: --relief-exaggeration must be positive. Using default value of 10.0.')
        args.relief_exaggeration = 10.0
    if args.surface and args.surface_exaggeration <= 0:
        print('Warning: --surface-exaggeration must be positive. Using default value of 1.0.')
        args.surface_exaggeration = 1.0
    if args.surface and args.relief:
        print('Warning: --surface and --relief were both provided. Using --surface.')
    
    # Ensure we have a filename
    if not args.filename:
        print('Error: No filename specified. Use -f or --filename to specify a NetCDF file.')
        return 1
    
    directory_setup(args.proj_name)

    plt.rcParams.update({'font.size': args.font_size, 'font.family': args.font_family})
    
    # Load boundary data if specified
    boundary_data = None
    if args.boundary:
        try:
            boundary_data = load_boundary_data(args.boundary)
        except Exception as e:
            print(f'Warning: Could not load boundary file {args.boundary}: {e}. Skipping boundary plotting.')
            boundary_data = None
    
    # Handle single vs multiple files
    if len(args.filename) == 1:
        # Single file mode (existing behavior)
        ds = xr.open_dataset(args.filename[0])
        if args.variable:
            variables = [ds[var] for var in args.variable]
        else:
            variables = list(ds.data_vars.values())
            if not variables:
                raise ValueError('No data variables found in the NetCDF file.')
        
        # Determine dataset/variable combinations for plotting
        n_vars = len(variables)
        
        # Create datasets list for consistent processing
        datasets = [ds] * n_vars
        file_variables = [(ds, var) for var in variables]
        
    else:
        # Multiple files mode (new feature)
        datasets = []
        file_variables = []
        
        for filename in args.filename:
            ds = xr.open_dataset(filename)
            datasets.append(ds)
            
            if args.variable:
                # Use specified variables (cycle through if fewer variables than files)
                var_name = args.variable[len(file_variables) % len(args.variable)]
                if var_name in ds.data_vars:
                    file_variables.append((ds, ds[var_name]))
                else:
                    print(f'Warning: Variable "{var_name}" not found in {filename}. Using first available variable.')
                    file_variables.append((ds, list(ds.data_vars.values())[0]))
            else:
                # Use first data variable from each file
                if len(ds.data_vars) == 0:
                    raise ValueError(f'No data variables found in {filename}.')
                file_variables.append((ds, list(ds.data_vars.values())[0]))
        
        # Determine subplot grid based on number of files
        n_files = len(args.filename)
        
    # Determine subplot layout
    def determine_layout(num_panels: int) -> tuple[int, int]:
        auto_layout = False
        nrows = args.nrows
        ncols = args.ncols

        try:
            if nrows is not None and nrows <= 0:
                raise ValueError('nrows must be positive')
            if ncols is not None and ncols <= 0:
                raise ValueError('ncols must be positive')

            if nrows is None and ncols is None:
                raise ValueError('No manual layout provided')

            if nrows is None:
                ncols = int(ncols)
                nrows = int(np.ceil(num_panels / ncols))
            elif ncols is None:
                nrows = int(nrows)
                ncols = int(np.ceil(num_panels / nrows))
            else:
                nrows = int(nrows)
                ncols = int(ncols)

            if nrows * ncols < num_panels:
                raise ValueError('nrows*ncols too small')

        except Exception as exc:
            if args.nrows is not None or args.ncols is not None:
                print(f'Warning: Invalid subplot layout ({exc}). Using automatic layout instead.')
            auto_layout = True

        if auto_layout:
            ncols = int(np.ceil(np.sqrt(num_panels)))
            nrows = int(np.ceil(num_panels / ncols))

        return nrows, ncols

    def get_cartopy_projection(projection_name: str):
        if not HAS_CARTOPY:
            return None
        projection_name = projection_name or 'PlateCarree'
        projection_map = {
            'PlateCarree': ccrs.PlateCarree(),
            'Robinson': ccrs.Robinson(),
            'Mollweide': ccrs.Mollweide(),
        }
        return projection_map.get(projection_name, ccrs.PlateCarree())

    total_panels = len(file_variables)
    nrows, ncols = determine_layout(total_panels)

    # Determine sharing parameters and subplot kwargs
    use_surface = bool(args.surface)
    use_cartopy = bool(args.global_plot and HAS_CARTOPY and not use_surface)
    if args.global_plot and not HAS_CARTOPY:
        print('Warning: --global-plot requested but Cartopy is not installed. Install cartopy to enable global plotting. Falling back to standard plotting.')
    if use_surface and args.global_plot:
        print('Warning: --global-plot is not supported with --surface. Ignoring global plot settings.')
    sharex = 'all' if args.sharex else False
    sharey = 'all' if args.sharey else False
    subplot_kwargs = {}
    projection = None
    data_crs = None
    global_cbar_orientation = args.global_cbar_orientation
    global_cbar_shrink = args.global_cbar_shrink
    global_cbar_pad = args.global_cbar_pad
    if use_surface:
        sharex = False
        sharey = False
        subplot_kwargs['subplot_kw'] = {'projection': '3d'}
    elif use_cartopy:
        projection = get_cartopy_projection(args.projection)
        if projection is None:
            use_cartopy = False
        else:
            sharex = False
            sharey = False
            subplot_kwargs['subplot_kw'] = {'projection': projection}
            data_crs = ccrs.PlateCarree()
            if global_cbar_shrink is None or global_cbar_shrink <= 0:
                print('Warning: global colorbar shrink must be positive. Using default value of 0.6.')
                global_cbar_shrink = 0.6
            if global_cbar_pad is None:
                global_cbar_pad = 0.05

    # Calculate figure size - ensure it's applied correctly even with cartopy
    fig_width = args.fig_size[0] * ncols
    fig_height = args.fig_size[1] * nrows
    
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharex=sharex,
        sharey=sharey,
        **subplot_kwargs,
    )
    
    # Ensure figure size is set (sometimes needed for cartopy projections)
    fig.set_size_inches(fig_width, fig_height)
    axes = np.atleast_2d(axes)
    
    shared_vmin = args.vmin
    shared_vmax = args.vmax
    if args.share_cbar and total_panels > 0:
        if shared_vmin is None or shared_vmax is None:
            panel_mins = []
            panel_maxs = []
            for _, var in file_variables:
                panel_data = var.values
                if args.unit is not None and args.unit != 'm':
                    units = var.attrs.get('units', '')
                    if units == 'meters' or units == 'm':
                        panel_data = panel_data * UNIT_CONVERSIONS[args.unit]
                finite_vals = panel_data[np.isfinite(panel_data)]
                if finite_vals.size:
                    panel_mins.append(float(np.nanmin(finite_vals)))
                    panel_maxs.append(float(np.nanmax(finite_vals)))
            if shared_vmin is None and panel_mins:
                shared_vmin = min(panel_mins)
            if shared_vmax is None and panel_maxs:
                shared_vmax = max(panel_maxs)

    shared_colorbar_label = None
    shared_cmap_name = args.cmap[0] if args.cmap else None
    skip_scalebar_warning = False
    skip_boundary_warning = False
    skip_surface_contour_warning = False
    visible_axes = []
    last_pcm = None
    for i, (ds, var) in enumerate(file_variables):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        visible_axes.append(ax)
        try:
            lon, lat = var.coords['lon'].values, var.coords['lat'].values
        except KeyError:
            lon, lat = var.coords['x'].values, var.coords['y'].values
        data = var.values

        units = var.attrs.get('units', '')

        # Convert units
        if args.unit is not None and args.unit != 'm':
            if units == 'meters' or units == 'm':
                data = data * UNIT_CONVERSIONS[args.unit]
                units = f'{args.unit}'

        # Select colormap for this subplot
        if len(args.cmap) == 1:
            cmap = get_colormap(args.cmap[0])
        else:
            # Use colormap corresponding to this file, cycling if fewer colormaps than files
            cmap_name = args.cmap[i % len(args.cmap)]
            cmap = get_colormap(cmap_name)

        plot_vmin = shared_vmin if args.share_cbar else args.vmin
        plot_vmax = shared_vmax if args.share_cbar else args.vmax

        if use_surface:
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                raise ValueError('Surface plotting requires at least one finite grid value.')
            surface_vmin = float(np.nanmin(finite)) if plot_vmin is None else float(plot_vmin)
            surface_vmax = float(np.nanmax(finite)) if plot_vmax is None else float(plot_vmax)
            if np.isclose(surface_vmin, surface_vmax):
                surface_vmax = surface_vmin + 1e-12
            surface_norm = Normalize(vmin=surface_vmin, vmax=surface_vmax)
            shaded_surface = np.nan_to_num(data, nan=surface_vmin)
            x_surface, y_surface, z_surface, _z_offset_km = build_surface_coordinates(
                lon=lon,
                lat=lat,
                data=data,
                exaggeration=args.surface_exaggeration,
            )
            facecolors = cmap(surface_norm(shaded_surface))
            facecolors[np.isnan(data), -1] = 0.0
            pcm = ScalarMappable(norm=surface_norm, cmap=cmap)
            pcm.set_array(data)
            ax.plot_surface(
                x_surface,
                y_surface,
                z_surface,
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            if args.contour:
                if not skip_surface_contour_warning:
                    print('Warning: --contour is not currently supported with --surface. Skipping contours.')
                    skip_surface_contour_warning = True
            ax.view_init(elev=args.surface_elev, azim=args.surface_azim)
            ax.set_box_aspect((
                np.ptp(x_surface) if x_surface.size > 1 else 1.0,
                np.ptp(y_surface) if y_surface.size > 1 else 1.0,
                max(np.ptp(z_surface), 1.0),
            ))
            ax.set_xlim(float(np.nanmin(x_surface)), float(np.nanmax(x_surface)))
            ax.set_ylim(float(np.nanmin(y_surface)), float(np.nanmax(y_surface)))
            if args.xlim:
                lon0 = float(np.nanmean(lon))
                lat0 = float(np.nanmean(lat))
                km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))
                ax.set_xlim((np.asarray(args.xlim, dtype=float) - lon0) * km_per_deg_lon)
            if args.ylim:
                lat0 = float(np.nanmean(lat))
                ax.set_ylim((np.asarray(args.ylim, dtype=float) - lat0) * 111.32)
            z_finite = z_surface[np.isfinite(z_surface)]
            if z_finite.size:
                ax.set_zlim(float(np.nanmin(z_finite)), float(np.nanmax(z_finite)))
            ax.set_axis_off()
        elif args.relief:
            relief_image, relief_norm, relief_origin = build_relief_image(
                lon=lon,
                lat=lat,
                data=data,
                cmap=cmap,
                vmin=plot_vmin,
                vmax=plot_vmax,
                azdeg=args.relief_azdeg,
                altdeg=args.relief_altdeg,
                exaggeration=args.relief_exaggeration,
            )
            extent = [
                float(np.nanmin(lon)),
                float(np.nanmax(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lat)),
            ]
            if use_cartopy:
                ax.imshow(
                    relief_image,
                    extent=extent,
                    origin=relief_origin,
                    transform=data_crs,
                    interpolation='nearest',
                )
            else:
                ax.imshow(
                    relief_image,
                    extent=extent,
                    origin=relief_origin,
                    interpolation='nearest',
                    aspect='auto',
                )
            pcm = ScalarMappable(norm=relief_norm, cmap=cmap)
            pcm.set_array(data)
        elif use_cartopy:
            pcm = ax.pcolormesh(
                lon,
                lat,
                data,
                cmap=cmap,
                shading='auto',
                vmin=plot_vmin,
                vmax=plot_vmax,
                transform=data_crs,
            )
        else:
            pcm = ax.pcolormesh(
                lon,
                lat,
                data,
                cmap=cmap,
                shading='auto',
                vmin=plot_vmin,
                vmax=plot_vmax,
            )
        last_pcm = pcm

        if args.contour and not use_surface:
            contour_kwargs = {
                'colors': args.contour_color,
                'linewidths': args.contour_linewidth,
                'levels': contour_levels if contour_levels is not None else 50,
                'alpha': args.contour_alpha,
            }
            if use_cartopy:
                ax.contour(
                    lon,
                    lat,
                    data,
                    transform=data_crs,
                    **contour_kwargs,
                )
            else:
                ax.contour(lon, lat, data, **contour_kwargs)

        # Set format_coord for status bar to show z value
        def make_format_coord(lon, lat, data) -> Callable:
            def format_coord(x, y) -> str:
                # Only show z if x and y are within the data bounds
                if (lon.min() <= x <= lon.max()) and (lat.min() <= y <= lat.max()):
                    ix = np.abs(lon - x).argmin()
                    iy = np.abs(lat - y).argmin()
                    try:
                        z = data[iy, ix]
                        return f"x={x:.2f}, y={y:.2f}, z={z:.2f}"
                    except Exception:
                        return f"x={x:.2f}, y={y:.2f}"
                else:
                    return f"x={x:.2f}, y={y:.2f}"
            return format_coord
        if not use_surface:
            ax.format_coord = make_format_coord(lon, lat, data)

        # Get long_name for use in title and colorbar
        long_name = var.attrs.get('long_name', var.name)

        # Select title for this subplot
        if args.title is None:
            title = long_name
        elif len(args.title) == 1:
            # Use single title for all subplots
            title = args.title[0]
        else:
            # Use title corresponding to this file, cycling if fewer titles than files
            title = args.title[i % len(args.title)]

        ax.set_title(title, fontweight='bold', fontsize=args.title_font_size)

        if use_surface:
            ax.grid(True, linewidth=0.3, alpha=0.4)
        elif use_cartopy:
            if args.xlim or args.ylim:
                lon_extent = args.xlim if args.xlim else (float(np.nanmin(lon)), float(np.nanmax(lon)))
                lat_extent = args.ylim if args.ylim else (float(np.nanmin(lat)), float(np.nanmax(lat)))
                ax.set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=data_crs)
            else:
                ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', alpha=0.6)
            ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', edgecolor='none', alpha=0.4)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black')
            # ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='dimgray')
            ax.gridlines(linestyle='--', linewidth=0.3, color='gray', alpha=0.5)
        else:
            ax.grid(which='both', linewidth=0.01)
            ax.minorticks_on()
            ax.grid(which='minor', linewidth=0.01)
            ax.set_xlim(args.xlim)
            ax.set_ylim(args.ylim)

        # Add boundary if specified
        if boundary_data is not None:
            boundary_kwargs = {'color': args.bound_color, 'linewidth': args.bound_linewidth}
            if use_surface:
                if not skip_boundary_warning:
                    print('Warning: --boundary is not currently supported with --surface. Skipping boundary overlay.')
                    skip_boundary_warning = True
            else:
                for boundary_lon, boundary_lat in boundary_data:
                    if use_cartopy:
                        ax.plot(boundary_lon, boundary_lat, transform=data_crs, **boundary_kwargs)
                    else:
                        ax.plot(boundary_lon, boundary_lat, **boundary_kwargs)

        colorbar_kwargs = {}
        if use_cartopy:
            colorbar_kwargs = {
                'orientation': global_cbar_orientation,
                'shrink': global_cbar_shrink,
                'pad': global_cbar_pad,
            }
        elif use_surface:
            colorbar_kwargs = {
                'orientation': args.cbar_orientation,
                'shrink': args.cbar_shrink,
                'pad': (
                    args.cbar_pad
                    if args.cbar_pad is not None
                    else (0.02 if args.cbar_orientation == 'horizontal' else 0.03)
                ),
            }
        else:
            colorbar_kwargs = {
                'orientation': args.cbar_orientation,
                'shrink': args.cbar_shrink,
                **({'pad': args.cbar_pad} if args.cbar_pad is not None else {}),
            }

        colorbar_label = None
        if args.cbar_title is not None:
            colorbar_label = f'{args.cbar_title} [{units}]' if units else args.cbar_title
        else:
            colorbar_label = f'{long_name} [{units}]' if units else long_name

        if args.share_cbar:
            if shared_colorbar_label is None:
                shared_colorbar_label = colorbar_label
            elif shared_colorbar_label != colorbar_label and args.cbar_title is None:
                shared_colorbar_label = 'Value'
            if len(args.cmap) > 1:
                cmap_name = args.cmap[i % len(args.cmap)]
                if cmap_name != shared_cmap_name:
                    shared_cmap_name = None
        else:
            if use_cartopy:
                cbar = fig.colorbar(pcm, ax=ax, **colorbar_kwargs)
                cbar.set_label(colorbar_label)
            else:
                add_panel_colorbar(fig, ax, pcm, colorbar_label, args, use_surface=use_surface)

        # Add scalebar
        if args.scalebar:
            if use_surface:
                if not skip_scalebar_warning:
                    print('Warning: Scalebar is not supported with --surface. Skipping scalebar.')
                    skip_scalebar_warning = True
            elif use_cartopy:
                if not skip_scalebar_warning:
                    print('Warning: Scalebar is not supported for projected global plots. Skipping scalebar.')
                    skip_scalebar_warning = True
            else:
                lon_range = args.xlim if args.xlim else (lon.min(), lon.max())
                scale_length = nice_scale_length(lon_range[1] - lon_range[0])
                if args.scalebar_units == 'km':
                    mean_lat = np.mean(lat) if args.ylim is None else np.mean(args.ylim)
                    scale_length_km = scale_length * 111.11 * np.cos(np.deg2rad(mean_lat))
                    scale_label = f'{int(scale_length_km)} km'
                else:
                    scale_label = f'{int(scale_length)} {args.scalebar_units}°'

                if args.scalebar_fancy:
                    # Create a segmented scalebar with alternating colors

                    n_segments = 4  # e.g., black, white, black
                    segment_length = scale_length / n_segments
                    colors = ['black', 'white'] * n_segments
                    colors = colors[:n_segments]  # Alternating colors
                    scale_bars = []

                    # Calculate segment width in axes coordinates (0 to 1)
                    lon_range = lon.max() - lon.min()
                    segment_width_axes = segment_length / lon_range  # Fraction of x-axis

                    for i in range(n_segments):
                        # Create a segment of the scalebar
                        sb = AnchoredSizeBar(
                            ax.transData,
                            segment_length,  # Length in data coordinates (degrees)
                            '',  # Label only on last segment
                            'lower left',  # Position
                            pad=0.35,
                            color=colors[i],
                            frameon=False,  # No frame for segments
                            size_vertical=0.04,  # Thickness
                            fontproperties={'size': 8},
                            alpha=0.25,
                            borderpad=0.5,
                            bbox_to_anchor=(i * segment_width_axes, 0),  # Offset in axes coordinates
                            bbox_transform=ax.transAxes  # Use axes coordinates for positioning
                        )
                        scale_bars.append(sb)
                        ax.add_artist(sb)

                        # Add centered label below the entire scalebar
                        ax.text(
                            0.6 * (n_segments * segment_width_axes),  # Center of scalebar
                            0.035,  # Lower left
                            scale_label,  # Label (e.g., "50°")
                            transform=ax.transAxes,
                            fontsize=8,
                            color='black',
                            ha='center',  # Center horizontally
                            va='top',  # Place below
                            bbox=dict(facecolor='none', edgecolor='none')
                        )
                else:
                    sb = AnchoredSizeBar(
                        ax.transData,
                        scale_length,  # Length in data coordinates (degrees)
                        scale_label,  # Label
                        'lower left',  # Position
                        pad=0.35,
                        color='black',
                        frameon=True,  # No frame for segments
                        size_vertical=0.04,  # Thickness
                        fontproperties={'size': 8},
                        alpha=0.25,
                        borderpad=0.5,
                    )
                    ax.add_artist(sb)
    
    # Hide unused subplots
    for i in range(len(file_variables), nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax.set_visible(False)

    if args.share_cbar and last_pcm is not None and visible_axes:
        if len(args.cmap) > 1 and shared_cmap_name is None:
            print('Warning: Multiple colormaps were supplied with --share-cbar. Using the last subplot colormap for the shared colorbar.')

        shared_colorbar_kwargs = {
            'orientation': args.shared_cbar_orientation,
            'shrink': args.shared_cbar_shrink,
            'pad': args.shared_cbar_pad,
        }
        cbar = fig.colorbar(last_pcm, ax=visible_axes, **shared_colorbar_kwargs)
        if shared_colorbar_label is not None:
            cbar.set_label(shared_colorbar_label, fontsize=args.shared_cbar_font_size)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(args.shared_cbar_font_size)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontsize(args.shared_cbar_font_size)
    
    if use_surface:
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.96, wspace=0.02, hspace=0.02)
    elif not args.share_cbar:
        plt.tight_layout()
    
    if args.save:
        figures_dir = Path(f'{args.proj_name}/results/figures')
        if len(args.filename) == 1:
            file_name = Path(args.filename[0]).stem
        else:
            # For multiple files, create a combined filename
            file_names = [Path(f).stem for f in args.filename]
            file_name = '_'.join(file_names)
        if use_surface:
            file_name = f'{file_name}_3D'
        output_path = figures_dir / f'{file_name}.png'
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        save_bbox = apply_save_pad(fig, renderer, tight_bbox, save_pad)
        plt.savefig(output_path, dpi=args.dpi, bbox_inches=save_bbox, pad_inches=0.0)
        print(f'Figure saved to: {output_path.absolute()}')
    else:
        plt.show()

if __name__ == '__main__':
    sys.exit(main())
