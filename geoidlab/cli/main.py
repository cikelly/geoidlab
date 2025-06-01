############################################################
# Main geoidlab CLI interface                              #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import shutil
from pathlib import Path
import glob

from geoidlab.cli.commands.reference import add_reference_arguments, main as ggm_main
from geoidlab.cli.commands.topo import add_topo_arguments, main as topo_main
from geoidlab.cli.commands.helmert import add_helmert_arguments, main as helmert_main
from geoidlab.cli.commands.plot import add_plot_arguments, main as plot_main
from geoidlab.cli.commands.geoid import add_geoid_arguments, main as geoid_main
from geoidlab.cli.commands.info import add_netcdf_info_arguments, main as netcdf_info_main
from geoidlab.cli.utils.config_parser import parse_config_file

def auto_visualize(args) -> None:
    """Run visualization on any NetCDF files in the results directory."""
    # Skip auto-visualization if this was already a viz command
    if getattr(args, 'subcommand', None) == 'viz':
        return
        
    # Check if save=True is set in config 
    if not getattr(args, 'save', True):  # Default to True if not set
        return
        
    # Get all NetCDF files in results directory
    results_dir = Path(getattr(args, 'proj_name', 'GeoidProject')) / 'results'
    if not results_dir.exists():
        return
        
    nc_files = list(results_dir.glob('*.nc'))
    if not nc_files:
        return
        
    # Clone the argparse namespace for viz command
    import copy
    viz_args = copy.deepcopy(args)
    viz_args.subcommand = 'viz'
    viz_args.save = True
    
    # Process each NetCDF file
    for nc_file in nc_files:
        viz_args.filename = str(nc_file)
        plot_main(viz_args)


class ConfigAction(argparse.Action):
    '''Custom action to handle --config/-c with or without a value.'''
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values if values is not None else '__COPY_TEMPLATE__')

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'GeoidLab: A toolkit for geodetic computations including gravity reductions, '
            'terrain quantities, GGM synthesis, geoid computation, and visualization.'
        ),
        epilog='Available commands: ggm, reduce, topo, viz, geoid, ncinfo'
    )
    from geoidlab.__version__ import __version__
    parser.add_argument('-v', '--version', action='version', version=f'geoidlab {__version__}')
    parser.add_argument('-c', '--config', nargs='?', default=None, action=ConfigAction, 
                        help='Path to configuration file (e.g., geoidlab.cfg). If not provided, copies from geoidlab/docs/geoidlab.cfg to the current directory.')
    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands', required=False)
    
    # GGM subcommand
    ggm_parser = subparsers.add_parser('ggm', help='Synthesize gravity field functionals from a global geopotential model (GGM)')
    add_reference_arguments(ggm_parser)
    ggm_parser.set_defaults(func=ggm_main)
    
    # Topo
    topo_parser = subparsers.add_parser('topo', help='Compute topographic quantities from a Digital Elevation Model (DEM)')
    add_topo_arguments(topo_parser)
    topo_parser.set_defaults(func=topo_main)
    
    # Helmert
    reduce_parser = subparsers.add_parser('reduce', help='Perform gravity reduction (Free-air, Bouguer, Helmert)')
    add_helmert_arguments(reduce_parser)
    reduce_parser.set_defaults(func=helmert_main)
    
    # Plot
    plot_parser = subparsers.add_parser('viz', help='Visualize data')
    add_plot_arguments(plot_parser)
    plot_parser.set_defaults(func=plot_main)
    
    # Geoid
    geoid_parser = subparsers.add_parser('geoid', help='Compute a geoid using the remove-compute-restore (RCR) method')
    add_geoid_arguments(geoid_parser)
    geoid_parser.set_defaults(func=geoid_main)
    
    # Info subcommand
    info_parser = subparsers.add_parser('ncinfo', 
                                        help='Print out information about a NetCDF file')
    add_netcdf_info_arguments(info_parser)
    info_parser.set_defaults(func=netcdf_info_main)
    
    args = parser.parse_args()
    
    # Handle config file
    if args.config is not None:
        if args.config == '__COPY_TEMPLATE__':
            main_dir = Path(__file__).parent.parent.parent 

            template_path = main_dir / 'docs' / 'geoidlab.cfg'
            dest_path = Path.cwd() / 'geoidlab.cfg'
            
            if not template_path.exists():
                print(
                    f'Error: Template config file {template_path} not found in geoidlab/docs.'
                    'Go to https://github.com/cikelly/geoidlab/tree/main/docs/ to download the template.'
                )
                sys.exit(1)
                
            if dest_path.exists():
                print(f"Note: '{dest_path}' already exists. Not overwriting.")
                print("To use the template, edit the existing geoidlab.cfg or specify a different config file with --config <path>.")
                sys.exit(0)
                
            try:
                shutil.copy(template_path, dest_path)
                print(f"Template configuration file copied to '{dest_path}'.")
                print("Please edit geoidlab.cfg and run `geoidlab --config geoidlab.cfg` to use it.")
                sys.exit(0)
            except (PermissionError, OSError) as e:
                print(f"Error: Failed to copy template config to '{dest_path}': {str(e)}")
                sys.exit(1)
        else:
            # Config file path provided, parse it
            args = parse_config_file(args.config, args)
            
    
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)
        
    # Run the command
    result = args.func(args)
    
    # Auto-visualize any output files if save=True
    auto_visualize(args)
    
    return result
    
    
if __name__ == '__main__':
    sys.exit(main())