############################################################
# Utitlity to parse config file                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import configparser
import os
import sys

from pathlib import Path

from geoidlab.cli.commands.reference import main as ggm_main
from geoidlab.cli.commands.topo import main as topo_main
from geoidlab.cli.commands.reduce import main as reduce_main
from geoidlab.cli.commands.plot import main as plot_main
from geoidlab.cli.commands.geoid import main as geoid_main
from geoidlab.cli.commands.info import main as netcdf_info_main
from geoidlab.cli.commands.prep import main as prep_main
from geoidlab.cli.commands.dtm import main as dtm_main

def parse_config_file(config_path: str, cli_args: argparse.Namespace) -> argparse.Namespace:
    '''
    Parse a geoidlab config file and merge with CLI arguments.
    
    Parameters
    ----------
    config_path: Path to the config file.
    cli_args   : Parsed CLI arguments (argparse.Namespace).
    
    Returns
    -------
    argparse.Namespace with merged arguments.
    '''
    config_path = Path(config_path).resolve()
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    
    config = configparser.ConfigParser(
        allow_no_value=True,
        comment_prefixes=('#', ';'),
        inline_comment_prefixes=('#', ';'),
    )
    config.read(config_path)
    
    # Initialize args with CLI arguments
    args_dict = vars(cli_args).copy()
    
    # Required subcommand
    if 'subcommand' not in config or 'command' not in config['subcommand']:
        print("Error: Config file must specify [subcommand] with 'command' (ggm, dtm, topo, reduce, viz, geoid, ncinfo).")
        sys.exit(1)
    
    subcommand = config['subcommand'].get('command', '').strip()
    # valid_subcommands = {'ggm', 'topo', 'reduce', 'viz', 'geoid', 'ncinfo'}
    valid_subcommands = {
        'ggm': ggm_main,
        'dtm': dtm_main,
        'topo': topo_main,
        'reduce': reduce_main,
        'viz': plot_main,
        'geoid': geoid_main,
        'ncinfo': netcdf_info_main,
        'prep': prep_main
    }
    # if subcommand not in valid_subcommands:
    #     print(f"Error: Invalid subcommand '{subcommand}'. Must be one of {valid_subcommands}.")
    #     sys.exit(1)
    if subcommand not in valid_subcommands:
        print(f"Error: Invalid subcommand '{subcommand}'. Must be one of {set(valid_subcommands.keys())}.")
        sys.exit(1)
    
    # Override subcommand if not set via CLI
    if not args_dict['subcommand']:
        args_dict['subcommand'] = subcommand
        args_dict['func'] = valid_subcommands[subcommand]
    elif args_dict['subcommand'] != subcommand:
        print(f"Warning: CLI subcommand '{args_dict['subcommand']}' overrides config subcommand '{subcommand}'.")
    
    # Parameter mappings: config key to CLI argument name
    param_map = {
        # input_data
        'input_file'            : ('input_file', None),
        'marine_data'           : ('marine_data', None),
        'marine_data_type'      : ('marine_data_type', 'free_air_anomaly'),
        'marine_tide_system'    : ('marine_tide_system', None),
        'residual_method'       : ('residual_method', 'station'),
        # ggm
        'model'                 : ('model', None),
        'model_dir'             : ('model_dir', None),
        'max_deg'               : ('max_deg', None),
        'icgem'                 : ('icgem', False),
        'dtm_model'             : ('dtm_model', None),
        'model_format'          : ('model_format', None),
        'earth2014_model'       : ('earth2014_model', None),
        'earth2014_resolution'  : ('earth2014_resolution', '5min'),
        'gravity_tide'          : ('gravity_tide', 'mean_tide'),
        'converted'             : ('converted', False),
        # grid
        'bbox'                  : ('bbox', None),
        'bbox_offset'           : ('bbox_offset', None),
        'grid_size'             : ('grid_size', None),
        'grid_unit'             : ('grid_unit', None),
        'grid_method'           : ('grid_method', None),
        # topography
        'topo'                  : ('topo', None),
        'topo_file'             : ('topo_file', None),
        'topo_url'              : ('topo_url', None),
        'topo_cog_url'          : ('topo_cog_url', None),
        'topo_lon_name'         : ('topo_lon_name', 'x'),
        'topo_lat_name'         : ('topo_lat_name', 'y'),
        'topo_height_name'      : ('topo_height_name', 'z'),
        'ref_topo'              : ('ref_topo', None),
        'dtm_nmax'              : ('dtm_nmax', None),
        'dtm_chunk_size'        : ('dtm_chunk_size', None),
        'chunk_memory_gb'       : ('chunk_memory_gb', None),
        'workers'               : ('workers', None),
        'leg_progress'          : ('leg_progress', False),
        'radius'                : ('radius', None),
        'ellipsoid'             : ('ellipsoid', None),
        'ellipsoid_name'        : ('ellipsoid_name', None),
        'interpolation_method'  : ('interpolation_method', None),
        'interp_method'         : ('interp_method', None),
        'tc_file'               : ('tc_file', None),
        'tc_grid_size'          : ('tc_grid_size', 30.0),
        'window_mode'           : ('window_mode', None),
        'approximation'          : ('approximation', False),
        'variable_density'      : ('variable_density', False),
        'density_model'         : ('density_model', '30s'),
        'density_resolution'    : ('density_resolution', None),
        'density_resolution_unit': ('density_resolution_unit', None),
        'density_file'          : ('density_file', None),
        'density_interp_method' : ('density_interp_method', 'nearest'),
        'density_unit'          : ('density_unit', 'kg/m3'),
        'density_save'          : ('density_save', True),
        # computation
        'do'                    : ('do', None),
        'start'                 : ('start', None),
        'end'                   : ('end', None),
        'parallel'              : ('parallel', False),
        'force_parallel'        : ('force_parallel', False),
        'force'                 : ('force', False),
        'threaded_legendre'     : ('threaded_legendre', False),
        'legendre_method'       : ('legendre_method', 'standard'),
        'chunk_size'            : ('chunk_size', None),
        'atm'                   : ('atm', False),
        'atm_method'            : ('atm_method', 'noaa'),
        'site'                  : ('site', False),
        'ellipsoidal_correction': ('ellipsoidal_correction', False),
        'apply_terrain_correction': ('apply_terrain_correction', True),
        'decimate'              : ('decimate', False),
        'decimate_threshold'    : ('decimate_threshold', None),
        # geoid
        'sph_cap'               : ('sph_cap', 1.0),
        'method'                : ('method', 'hg'),
        'ind_grid_size'         : ('ind_grid_size', 30.0),
        'target_tide_system'    : ('target_tide_system', 'tide_free'),
        # project
        'proj_name'             : ('proj_name', 'GeoidProject'),
        # viz
        'filename'              : ('filename', None),
        'variable'              : ('variable', None),
        'cmap'                  : ('cmap', None),
        'fig_size'              : ('fig_size', None),
        'vmin'                  : ('vmin', None),
        'vmax'                  : ('vmax', None),
        'font_size'             : ('font_size', None),
        'title'                 : ('title', None),
        'title_font_size'       : ('title_font_size', None),
        'font_family'           : ('font_family', None),
        'cbar_title'            : ('cbar_title', None),
        'cbar_orientation'      : ('cbar_orientation', 'vertical'),
        'cbar_shrink'           : ('cbar_shrink', 1.0),
        'cbar_pad'              : ('cbar_pad', None),
        'cbar_location'         : ('cbar_location', 'right'),
        'list_cmaps'            : ('list_cmaps', False),
        'save'                  : ('save', False),
        'dpi'                   : ('dpi', None),
        'save_pad'              : ('save_pad', None),
        'xlim'                  : ('xlim', None),
        'ylim'                  : ('ylim', None),
        'scalebar'              : ('scalebar', False),
        'scalebar_units'        : ('scalebar_units', None),
        'scalebar_fancy'        : ('scalebar_fancy', False),
        'unit'                  : ('unit', None),
        'relief'                : ('relief', False),
        'relief_exaggeration'   : ('relief_exaggeration', 10.0),
        'relief_azdeg'          : ('relief_azdeg', 135.0),
        'relief_altdeg'         : ('relief_altdeg', 45.0),
        'surface'               : ('surface', False),
        'surface_exaggeration'  : ('surface_exaggeration', 0.5),
        'surface_elev'          : ('surface_elev', 30.0),
        'surface_azim'          : ('surface_azim', -110.0),
        'boundary'              : ('boundary', None),
        'bound_color'           : ('bound_color', 'k'),
        'bound_linewidth'       : ('bound_linewidth', 1.2),
        'sharex'                : ('sharex', False),
        'sharey'                : ('sharey', False),
        'nrows'                 : ('nrows', None),
        'ncols'                 : ('ncols', None),
        'global_plot'           : ('global_plot', False),
        'projection'            : ('projection', 'PlateCarree'),
        'global_cbar_orientation': ('global_cbar_orientation', 'horizontal'),
        'global_cbar_shrink'    : ('global_cbar_shrink', 0.6),
        'global_cbar_pad'       : ('global_cbar_pad', 0.05),
        'share_cbar'            : ('share_cbar', False),
        'shared_cbar_orientation': ('shared_cbar_orientation', 'vertical'),
        'shared_cbar_shrink'    : ('shared_cbar_shrink', 0.5),
        'shared_cbar_pad'       : ('shared_cbar_pad', 0.02),
        'shared_cbar_font_size' : ('shared_cbar_font_size', 12),
        'contour'               : ('contour', False),
        'contour_color'         : ('contour_color', 'black'),
        'contour_linewidth'     : ('contour_linewidth', 0.25),
        'contour_alpha'         : ('contour_alpha', 0.8),
        'contour_levels'        : ('contour_levels', None),
        # ncinfo
        'verbose'               : ('verbose', False),
    }
    
    # Type conversion rules
    def convert_value(key: str, value: str, config_dir: Path) -> any:
        if not value.strip():
            return None
        if key in {'parallel', 'force_parallel', 'force', 'threaded_legendre', 'icgem', 'converted', 'atm', 'decimate', 'save', 'scalebar', 'scalebar_fancy', 'verbose', 'site', 'ellipsoidal_correction', 'approximation', 'variable_density', 'density_save', 'apply_terrain_correction', 'leg_progress', 'list_cmaps', 'relief', 'surface', 'sharex', 'sharey', 'global_plot', 'share_cbar', 'contour'}:
            return value.lower() in {'true', 'yes', '1'}
        if key in {'max_deg', 'chunk_size', 'decimate_threshold', 'font_size', 'title_font_size', 'dpi', 'dtm_nmax', 'dtm_chunk_size', 'density_resolution', 'workers', 'nrows', 'ncols', 'shared_cbar_font_size'}:
            return int(value)
        if key in {'radius', 'bbox_offset', 'grid_size', 'sph_cap', 'tc_grid_size', 'ind_grid_size', 'vmin', 'vmax', 'chunk_memory_gb', 'cbar_shrink', 'cbar_pad', 'relief_exaggeration', 'relief_azdeg', 'relief_altdeg', 'surface_exaggeration', 'surface_elev', 'surface_azim', 'bound_linewidth', 'global_cbar_shrink', 'global_cbar_pad', 'shared_cbar_shrink', 'shared_cbar_pad', 'contour_linewidth', 'contour_alpha'}:
            return float(value)
        if key in {'bbox', 'fig_size', 'xlim', 'ylim'}:
            return [float(x) for x in value.split()]
        if key == 'save_pad':
            return value.split()
        if key == 'variable':
            return [x.strip() for x in value.split(',')]
        if key in {'input_file', 'marine_data', 'tc_file', 'ref_topo', 'filename', 'model_dir', 'density_file', 'topo_file', 'boundary'}:
            # Resolve relative paths relative to config file directory
            path = Path(value)
            if not path.is_absolute():
                path = config_dir / path
            return str(path.resolve())
        return value
    
    # Process each section
    config_dir = config_path.parent
    for section in config.sections():
        if section == 'subcommand':
            continue
        
        for key, value in config[section].items():
            if key not in param_map:
                print(f"Warning: Ignoring unknown parameter '{key}' in section [{section}].")
                continue
            cli_key, default = param_map[key]
            # Set value if not already set via CLI
            if args_dict.get(cli_key) is None:
                args_dict[cli_key] = convert_value(key, value, config_dir) if value.strip() else default
                
    
    # Set defaults for any unset parameters
    for key, (cli_key, default) in param_map.items():
        if args_dict.get(cli_key) is None:
            args_dict[cli_key] = default
            
    
    # Validation for required parameters
    if args_dict['subcommand'] == 'geoid':
        if not args_dict.get('input_file'):
            print("Error: 'input_file' is required for geoid subcommand.")
            sys.exit(1)
        # Verify input_file exists
        input_file = args_dict.get('input_file')
        if input_file and not Path(input_file).exists():
            print(f"Error: Input file '{input_file}' does not exist.")
            sys.exit(1)
    elif args_dict['subcommand'] == 'topo':
        topo_sources = [
            args_dict.get('topo'),
            args_dict.get('topo_file'),
            args_dict.get('topo_url'),
            args_dict.get('topo_cog_url'),
        ]
        if sum(source is not None for source in topo_sources) != 1:
            print("Error: specify exactly one DEM source for topo: 'topo', 'topo_file', 'topo_url', or 'topo_cog_url'.")
            sys.exit(1)
        if not args_dict.get('bbox'):
            print("Error: 'bbox' is required for topo subcommand.")
            sys.exit(1)
    elif args_dict['subcommand'] == 'reduce':
        if not args_dict.get('input_file'):
            print("Error: 'input_file' is required for reduce subcommand.")
            sys.exit(1)
    elif args_dict['subcommand'] == 'ggm':
        if not args_dict.get('model'):
            print("Error: 'model' is required for ggm subcommand.")
            sys.exit(1)
    elif args_dict['subcommand'] == 'viz':
        if not args_dict.get('filename'):
            print("Error: 'filename' is required for viz subcommand.")
            sys.exit(1)
    elif args_dict['subcommand'] == 'ncinfo':
        if not args_dict.get('filename'):
            print("Error: 'filename' is required for ncinfo subcommand.")
            sys.exit(1)

    if args_dict['subcommand'] == 'viz':
        for key in {'filename', 'cmap', 'title'}:
            if isinstance(args_dict.get(key), str):
                args_dict[key] = [args_dict[key]]
        if isinstance(args_dict.get('variable'), str):
            args_dict['variable'] = [args_dict['variable']]
    
    # Convert grid flag for reduce
    if args_dict['subcommand'] == 'reduce' and args_dict.get('bbox') and args_dict.get('grid_size'):
        args_dict['grid'] = True
    
    return argparse.Namespace(**args_dict)
