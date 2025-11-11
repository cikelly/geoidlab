import argparse
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from typing import Callable
from pathlib import Path
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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
    parser.add_argument('--list-cmaps', action='store_true', help='List available colormaps and exit')
    parser.add_argument('--save', action='store_true', help='Save figure')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saving figure')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject', help='Name of the project')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help='X-axis limits')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='Y-axis limits')
    parser.add_argument('--scalebar', action='store_true', help='Show scalebar')
    parser.add_argument('--scalebar-units', type=str, default='km', choices=['km', 'degrees'], help='Scalebar units')
    parser.add_argument('--scalebar-fancy', action='store_true', help='Use fancy scalebar')
    parser.add_argument('-u', '--unit', type=str, default=None, choices=['m', 'cm', 'mm'], help='Unit to display data with length units')
    parser.add_argument('--boundary', type=str, help='CSV file containing boundary coordinates (columns: lon, lat)')
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

def main(args=None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description='Plot a NetCDF file')
        add_plot_arguments(parser)
        args = parser.parse_args()
        
    if getattr(args, 'list_cmaps', False):
        list_colormaps()
        return 0
    
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
            boundary_data = pd.read_csv(args.boundary)
            if 'lon' not in boundary_data.columns or 'lat' not in boundary_data.columns:
                print(f'Warning: Boundary file {args.boundary} must contain "lon" and "lat" columns. Skipping boundary plotting.')
                boundary_data = None
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
    use_cartopy = bool(args.global_plot and HAS_CARTOPY)
    if args.global_plot and not HAS_CARTOPY:
        print('Warning: --global-plot requested but Cartopy is not installed. Install cartopy to enable global plotting. Falling back to standard plotting.')
    sharex = 'all' if args.sharex else False
    sharey = 'all' if args.sharey else False
    subplot_kwargs = {}
    projection = None
    data_crs = None
    global_cbar_orientation = args.global_cbar_orientation
    global_cbar_shrink = args.global_cbar_shrink
    global_cbar_pad = args.global_cbar_pad
    if use_cartopy:
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
    
    skip_scalebar_warning = False
    for i, (ds, var) in enumerate(file_variables):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
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

        if use_cartopy:
            pcm = ax.pcolormesh(
                lon,
                lat,
                data,
                cmap=cmap,
                shading='auto',
                vmin=args.vmin,
                vmax=args.vmax,
                transform=data_crs,
            )
        else:
            pcm = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto', vmin=args.vmin, vmax=args.vmax)

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

        if use_cartopy:
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
            if use_cartopy:
                ax.plot(boundary_data.lon, boundary_data.lat, transform=data_crs, **boundary_kwargs)
            else:
                ax.plot(boundary_data.lon, boundary_data.lat, **boundary_kwargs)

        colorbar_kwargs = {}
        if use_cartopy:
            colorbar_kwargs = {
                'orientation': global_cbar_orientation,
                'shrink': global_cbar_shrink,
                'pad': global_cbar_pad,
            }

        colorbar_label = None
        if args.cbar_title is not None:
            colorbar_label = f'{args.cbar_title} [{units}]' if units else args.cbar_title
        else:
            colorbar_label = f'{long_name} [{units}]' if units else long_name

        cbar = fig.colorbar(pcm, ax=ax, **colorbar_kwargs)
        cbar.set_label(colorbar_label)

        # Add scalebar
        if args.scalebar:
            if use_cartopy:
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
    
    plt.tight_layout()
    
    if args.save:
        figures_dir = Path(f'{args.proj_name}/results/figures')
        if len(args.filename) == 1:
            file_name = args.filename[0].split('/')[-1].split('.')[0]
        else:
            # For multiple files, create a combined filename
            file_names = [f.split('/')[-1].split('.')[0] for f in args.filename]
            file_name = '_'.join(file_names)
        output_path = figures_dir / f'{file_name}.png'
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f'Figure saved to: {output_path.absolute()}')
    else:
        plt.show()

if __name__ == '__main__':
    sys.exit(main())
