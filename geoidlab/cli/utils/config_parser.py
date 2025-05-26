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
from geoidlab.cli.commands.faye import main as faye_main
from geoidlab.cli.commands.plot import main as plot_main
from geoidlab.cli.commands.geoid import main as geoid_main
from geoidlab.cli.commands.info import main as netcdf_info_main

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
        print("Error: Config file must specify [subcommand] with 'command' (ggm, topo, reduce, viz, geoid, ncinfo).")
        sys.exit(1)
    
    subcommand = config['subcommand'].get('command', '').strip()
    # valid_subcommands = {'ggm', 'topo', 'reduce', 'viz', 'geoid', 'ncinfo'}
    valid_subcommands = {
        'ggm': ggm_main,
        'topo': topo_main,
        'reduce': faye_main,
        'viz': plot_main,
        'geoid': geoid_main,
        'ncinfo': netcdf_info_main
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
        'input_file'          : ('input_file', None),
        'marine_data'         : ('marine_data', None),
        # ggm
        'model'               : ('model', None),
        'model_dir'           : ('model_dir', None),
        'max_deg'             : ('max_deg', None),
        'icgem'               : ('icgem', False),
        'gravity_tide'        : ('gravity_tide', 'mean_tide'),
        'converted'           : ('converted', False),
        # grid
        'bbox'                : ('bbox', None),
        'bbox_offset'         : ('bbox_offset', None),
        'grid_size'           : ('grid_size', None),
        'grid_unit'           : ('grid_unit', None),
        'grid_method'         : ('grid_method', None),
        # topography
        'topo'                : ('topo', None),
        'ref_topo'            : ('ref_topo', None),
        'radius'              : ('radius', None),
        'ellipsoid'           : ('ellipsoid', None),
        'interpolation_method': ('interpolation_method', None),
        'interp_method'       : ('interp_method', None),
        'tc_file'             : ('tc_file', None),
        'tc_grid_size'        : ('tc_grid_size', 30.0),
        'window_mode'         : ('window_mode', None),
        # computation
        'do'                  : ('do', None),
        'start'               : ('start', None),
        'end'                 : ('end', None),
        'parallel'            : ('parallel', False),
        'chunk_size'          : ('chunk_size', None),
        'atm'                 : ('atm', False),
        'atm_method'          : ('atm_method', 'noaa'),
        'decimate'            : ('decimate', False),
        'decimate_threshold'  : ('decimate_threshold', None),
        # geoid
        'sph_cap'             : ('sph_cap', 1.0),
        'method'              : ('method', 'hg'),
        'ind_grid_size'       : ('ind_grid_size', 30.0),
        'target_tide_system'  : ('target_tide_system', 'tide_free'),
        # viz
        'filename'            : ('filename', None),
        'variable'            : ('variable', None),
        'cmap'                : ('cmap', None),
        'fig_size'            : ('fig_size', None),
        'vmin'                : ('vmin', None),
        'vmax'                : ('vmax', None),
        'font_size'           : ('font_size', None),
        'title_font_size'     : ('title_font_size', None),
        'font_family'         : ('font_family', None),
        'save'                : ('save', False),
        'dpi'                 : ('dpi', None),
        'proj_name'           : ('proj_name', None),
        'xlim'                : ('xlim', None),
        'ylim'                : ('ylim', None),
        'scalebar'            : ('scalebar', False),
        'scalebar_units'      : ('scalebar_units', None),
        'scalebar_fancy'      : ('scalebar_fancy', False),
        'unit'                : ('unit', None),
        # ncinfo
        'filename'            : ('filename', None),  # Used for viz, ncinfo
        'proj_name'           : ('proj_name', 'GeoidProject'),
        'verbose'             : ('verbose', False),
    }
    
    # Type conversion rules
    def convert_value(key: str, value: str, config_dir: Path) -> any:
        if not value.strip():
            return None
        if key in {'parallel', 'icgem', 'converted', 'atm', 'decimate', 'save', 'scalebar', 'scalebar_fancy'}:
            return value.lower() in {'true', 'yes', '1'}
        if key in {'max_deg', 'chunk_size', 'decimate_threshold', 'font_size', 'title_font_size', 'dpi'}:
            return int(value)
        if key in {'radius', 'bbox_offset', 'grid_size', 'sph_cap', 'tc_grid_size', 'ind_grid_size', 'vmin', 'vmax'}:
            return float(value)
        if key in {'bbox', 'fig_size', 'xlim', 'ylim'}:
            return [float(x) for x in value.split()]
        if key == 'variable':
            return [x.strip() for x in value.split(',')]
        if key in {'input_file', 'marine_data', 'tc_file', 'ref_topo', 'filename', 'model_dir'}:
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
        if not args_dict.get('topo'):
            print("Error: 'topo' is required for topo subcommand.")
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
    
    # Convert grid flag for reduce
    if args_dict['subcommand'] == 'reduce' and args_dict.get('bbox') and args_dict.get('grid_size'):
        args_dict['grid'] = True
    
    return argparse.Namespace(**args_dict)