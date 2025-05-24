############################################################
# CLI for printing NetCDF file information                 #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys

import xarray as xr
from pathlib import Path


def add_netcdf_info_arguments(parser: argparse.ArgumentParser) -> None:
    '''Add arguments for printing information about a NetCDF file.'''
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Path to the NetCDF file to inspect.')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject',
                        help='Name of the project directory. Defaults to "GeoidProject.')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed information about variables.')

def main(args=None) -> None:
    '''Main function to print information about a NetCDF file.'''
    if args is None:
        parser = argparse.ArgumentParser(
            description='Print information about a NetCDF file.',
        )
        add_netcdf_info_arguments(parser)
        args = parser.parse_args()
    
    # Handle both absolute and project-relative paths
    filepath = Path(args.file)
    # print(filepath)
    # if not filepath.is_absolute():
    #     filepath = Path(args.proj_name) / Path('results') / filepath
    # if not filepath.is_absolute():
    #     filepath = Path(args.proj_name) / filepath
    
    # print(filepath)
    
    if not filepath.exists():
        print(f'Error: The file {filepath} does not exist.')
        return
    
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        print(f'Error opening file {filepath}: {e}')
        return 1
    
    # Print basic information
    print('\nFile Information:')
    print(f'---------------')
    print(f'Path: {filepath}')
    print(f'Size: {filepath.stat().st_size / 1024:.1f} KB')
    
    print('\nDimensions:')
    print('-----------')
    for dim, size in ds.dims.items():
        print(f'{dim}: {size}')

    print('\nVariables:')
    print('----------')
    for var_name, var in ds.data_vars.items():
        print(f'\n{var_name}:')
        print(f'  Shape: {var.shape}')
        print(f'  Dtype: {var.dtype}')
        if var.attrs:
            print('  Attributes:')
            for key, value in var.attrs.items():
                print(f'    {key}: {value}')

    print('\nGlobal Attributes:')
    print('-----------------')
    for key, value in ds.attrs.items():
        print(f'{key}: {value}')

    if args.verbose:
        print('\nCoordinates:')
        print('------------')
        if not ds.dims:
            print('No dimensions found in dataset.')
        else:
            for coord_name, coord in ds.coords.items():
                # Skip coordinates that are not dimensions
                if coord_name not in ds.dims:
                    continue
                print(f'\n{coord_name}:')
                try:
                    flat_values = coord.values.flatten()
                    if len(flat_values) > 1:
                        print(f'  Values: {flat_values[0]}...{flat_values[-1]}')
                    elif len(flat_values) == 1:
                        print(f'  Values: {flat_values[0]}')
                    else:
                        print(f'  Values: Empty array')
                except Exception as e:
                    print(f'  Values: Unable to access values - {e}')
                if coord.attrs:
                    print('  Attributes:')
                    for key, value in coord.attrs.items():
                        print(f'    {key}: {value}')

    ds.close()
    return 0

if __name__ == '__main__':
    sys.exist(main())