############################################################
# Geoid workflow CLI interface                             #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path

from geoidlab.cli.commands.utils.common import (
    validate_params, 
    directory_setup,
    get_grid_lon_lat
)
from geoidlab.icgem import download_ggm, get_ggm_tide_system
from geoidlab.utils.io import save_to_netcdf

def download(model: str, model_dir: str | Path = None) -> dict:
    '''
    Download the Global Geopotential Model (GGM) to the specificed directory
    
    Parameters
    ----------
    model     : Name of the GGM (with or without .gfc extension)
    model_dir : Directory to download GGM to (e.g., <proj-name>/downloads)
    
    Returns
    -------
    dict      : Dictionary with status and output file path
    '''
    model_dir = Path('.') if model_dir is None else Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = (model_dir / model).with_suffix('.gfc')

    if not model_path.exists():
        download_ggm(model, model_dir) 

    return {'status': 'success', 'output_file': str(model_path)}

def compute_gravity_anomaly(
    max_deg: int, 
    model: str,
    lonlatheight: pd.DataFrame | np.ndarray = None,
    model_dir: str | Path = None,
    chunk_size: int = 500,
    parallel: bool = False,
    ellipsoid: str = 'wgs84',
    output_dir: str | Path = 'results',
    tide_system: str | None = None,
    converted: bool = False,
    input_file: str = None,
    bbox: list = [None, None, None, None],
    bbox_offset: float = 1.,
    grid_size: float = None,
    grid_unit: str = 'minutes'
) -> dict:
    '''
    Compute gravity anomalies for geoid computation
    
    Parameters
    ----------
    max_deg     : Maximum degree of truncation for the computations
    model       : Name of the GGM (with or without .gfc extension)
    lonlatheight: DataFrame with columns lon, lat, and height or Numpy array with three columns
    model_dir   : Directory to download GGM to
    chunk_size  : Chunk size for parallel processing
    parallel    : Enable parallel processing
    ellipsoid   : Reference ellipsoid
    output_dir  : Directory to write outputs to
    tide_system : Tide system of the surface gravity data (Options: mean_tide, zero_tide, tide_free)
    converted   : Whether input data is already in the target tide system
    input_file  : Path to the input file (for naming converted output)
    bbox        : Bounding box [W,S,E,N] of the study area
    bbox_off    : Offset from the bounding box (in degrees)
    grid_size   : Grid size in minutes (e.g., 5 for a 5-by-5 minute grid)
    grid_unit   : Grid unit (e.g., 'minutes')
    
    Returns
    -------
    dict        : Dictionary with status and output file path
    path (if applicable)
    '''
    from geoidlab.ggm import GlobalGeopotentialModel
    from geoidlab.tide import GravityTideSystemConverter
    
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'Dg_ggm.csv' if lonlatheight is not None else output_dir / 'Dg_ggm.nc'
    
    print(f'\nGravity anomalies will be computed with max_deg={max_deg}, ellipsoid={ellipsoid}')
    
    # Directory setup
    model_dir = Path('.') if model_dir is None else Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = (model_dir / model).with_suffix('.gfc')
    
    # Convert lonlatheight to DataFrame if necessary
    if lonlatheight is not None:
        if isinstance(lonlatheight, np.ndarray):
            lonlatheight = pd.DataFrame(lonlatheight, columns=['lon', 'lat', 'height'])
        
        # Handle tide system conversion
        converted_data_path = None
        if not converted and tide_system is not None:
            valid_tide_systems = ['mean_tide', 'zero_tide', 'tide_free']
            if tide_system not in valid_tide_systems:
                raise ValueError(f'tide_system must be one of: {valid_tide_systems}')
            
            # Get GGM tide system
            ggm_tide = get_ggm_tide_system(icgem_file=model_path, model_dir=model_dir)
            if ggm_tide not in valid_tide_systems:
                raise ValueError(f'GGM tide system {ggm_tide} is not supported.')
            
            if ggm_tide != tide_system:
                print(f'Converting input data from {tide_system} to {ggm_tide} system...')
                
                # Define conversion method dispatch table
                conversion_methods = {
                    ('mean_tide', 'tide_free'): 'mean2free',
                    ('tide_free', 'mean_tide'): 'free2mean',
                    ('mean_tide', 'zero_tide'): 'mean2zero',
                    ('zero_tide', 'mean_tide'): 'zero2mean',
                    ('zero_tide', 'tide_free'): 'zero2free',
                    ('tide_free', 'zero_tide'): 'free2zero'
                }
                # Get the conversion method
                conversion_key = (tide_system, ggm_tide)
                if conversion_key not in conversion_methods:
                    raise ValueError(f'No conversion available from {tide_system} to {ggm_tide}')
                
                if 'gravity' not in lonlatheight.columns:
                    print('Warning: No gravity data provided; using placeholder values for tide conversion')
                    lonlatheight['gravity'] = 980_000
                
                # Perform conversion
                converter = GravityTideSystemConverter(data=lonlatheight)
                conversion_method = getattr(converter, conversion_methods[conversion_key])
                converted_data = conversion_method()
                
                # Map GGM tide system to column suffix
                tide_suffix_map = {
                    'tide_free': 'free',
                    'mean_tide': 'mean',
                    'zero_tide': 'zero'
                }
                
                target_suffix = tide_suffix_map[ggm_tide]
                
                # Update lonlatheight with converted values. Overwrite the original values
                lonlatheight[f'height'] = converted_data[f'height_{target_suffix}']
                if 'gravity' in converted_data.columns:
                    lonlatheight['gravity'] = converted_data[f'g_{target_suffix}']
                
                # Save converted data
                input_filename = Path(input_file).stem if input_file else 'lonlatheight'
                converted_data_path = output_dir / f'{input_filename}_{ggm_tide}.csv'
                lonlatheight.to_csv(converted_data_path, index=False)
                print(f'Converted data saved to {converted_data_path}')
    else:
        if bbox is None:
            raise ValueError('bbox must be provided if input_file is not provided')
        if grid_size is None:
            raise ValueError('grid_size must be provided if input_file is not provided')

        min_lon, max_lon, min_lat, max_lat = bbox
        # Update min_lon, min_lat, max_lon, max_lat based on bbox_offset
        min_lon -= bbox_offset
        min_lat -= bbox_offset
        max_lon += bbox_offset
        max_lat += bbox_offset
        grid_extent = (min_lon, max_lon, min_lat, max_lat)
        
        lon_grid, lat_grid = get_grid_lon_lat(grid_extent, grid_size, grid_unit)
        
        lonlatheight = pd.DataFrame({
            'lon': lon_grid.flatten(),
            'lat': lat_grid.flatten(),
            'height': 0,
            'gravity': 0
        }) 

    # Compute gravity anomalies
    print(f'Computing gravity anomalies...')
    
    model = GlobalGeopotentialModel(
        model_name=model_path.stem, 
        grav_data=lonlatheight,
        nmax=max_deg,
        zonal_harmonics=True, 
        ellipsoid=ellipsoid,
        model_dir=model_dir,
        chunk_size=chunk_size
    )
    
    Dg_ggm = model.gravity_anomaly(parallel=parallel)
    
    print(f'Processing complete. Writing to {output_file}...')
    
    if input_file:
        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'lon': lonlatheight['lon'],
            'lat': lonlatheight['lat'],
            'Dg_ggm': Dg_ggm
        })
        
        df.to_csv(output_file, index=False)
    else:
        Dg_ggm = Dg_ggm.reshape(lon_grid.shape)
        
        status = save_to_netcdf(
            data=Dg_ggm,
            lon=lon_grid,
            lat=lat_grid,
            dataset_key='Dg_ggm',
            filepath=output_file
        )

    if status == 'Success':
        print(f'Reference gravity anomalies written to {output_file}\n')
    else:
        print(f'Failed to write reference gravity anomalies to {output_file}\n')
    
    return {'status': 'success', 'output_file': str(output_file)}
    

def compute_geoid(
    max_deg, 
    model,
    model_dir: str | Path = 'downloads',
    chunk_size: int = 500,
    parallel: bool = True,
    ellipsoid: str = 'wgs84',
    output_dir: str | Path = 'results',
    tide_system: str | None = None,
    converted: bool = False,
    input_file: str = None,
    bbox: list = [None, None, None, None],
    bbox_offset: float = 1.,
    grid_size: float = None,
    grid_unit: str = 'minutes'
) -> dict:
    '''
    Compute reference geoid heights (N_ref) on the final grid
    
    Parameters
    ----------
    max_deg    : Maximum degree of truncation for the computations
    model      : Name of the GGM (with or without .gfc extension)
    model_dir  : Directory to download GGM to
    chunk_size : Chunk size for parallel processing
    parallel   : Enable parallel processing
    ellipsoid  : Reference ellipsoid
    output_dir : Directory to write outputs to
    tide_system: Name of the tide system to use for conversion
    converted  : Whether to use converted data
    input_file : Input file to use for conversion
    bbox       : Bounding box to use for conversion
    bbox_offset: Offset from the bounding box (in degrees)
    grid_size  : Grid size in degrees (e.g., 5 for a 5-degree grid)
    grid_unit  : Unit of the grid size (e.g., 'degrees')
    
    
    Returns
    -------
    dict      : Dictionary with status and output file path
    '''
    print(f'Reference geoid heights will be computed with max_deg={max_deg}, ellipsoid={ellipsoid}')
    
    output_dir = Path(output_dir) / 'N_ref.nc'
    return {'status': 'success', 'output_file': 'N_ref.nc'}


def compute(
    max_deg, 
    model,
    lonlatheight: pd.DataFrame | np.ndarray | None = None,
    model_dir: str | Path = 'GeoidProject/downloads',
    chunk_size: int = 500,
    parallel: bool = True,
    ellipsoid: str = 'wgs84',
    output_dir: str|Path = 'results',
    do: str = 'all',
    start: str = None,
    end: str = None,
    tide_system: str | None = None,
    converted: bool = False,
    input_file: str = None,
    bbox: list = [None, None, None, None],
    bbox_offset: float = 1.,
    grid_size: float = None,
    grid_unit: str = 'minutes'
) -> None:
    '''
    Compute reference gravity anomalies and/or geoid heights for geoid computation
    
    Parameters
    ----------
    max_deg     : Maximum degree of truncation for the computations
    model       : Name of the GGM (with or without .gfc extension)
    lonlatheight: DataFrame with columns lon, lat, and height or Numpy array with three columns
    model_dir   : Directory to download GGM to
    chunk_size  : Chunk size for parallel processing
    parallel    : Enable parallel processing
    ellipsoid   : Reference ellipsoid
    output_dir  : Directory to write outputs to
    do          : Computation steps to perform ('download', 'gravity-anomaly', 'reference-geoid)
    start       : First step to perform
    end         : Last step to perform
    tide_system : Tide system of the surface gravity data (Options: mean_tide, zero_tide, tide_free)
    converted   : Whether input data is already in the target tide system
    input_file  : Path to the input file (for naming converted output)
    
    Returns
    -------
    dict        : Dictionary with status and output file(s)
    '''
    results = {}
    
    # Define the workflow order
    workflow = ['download', 'gravity-anomaly', 'reference-geoid']
    
    if do == 'download' or do == 'all':
        results['download'] = download(model, model_dir)
    
    # Determine tasks to run
    tasks = []
    if do != 'all' and (start or end):
        raise ValueError('Cannot specify both --do and --start/--end')
    
    if do == 'all':
        tasks = workflow
    elif start or end:
        start_idx = 0 if start is None else workflow.index(start)
        end_idx = len(workflow) - 1 if end is None else workflow.index(end)
        if start_idx > end_idx:
            raise ValueError('Start task must come before end task in workflow')
        tasks = workflow[start_idx:end_idx+1]
    else:
        tasks = [do]
    
    # Execute tasks
    for task in tasks:
        if task == 'download':
            results['download'] = download(model, model_dir)
        elif task == 'gravity-anomaly':
            results['gravity-anomaly'] = compute_gravity_anomaly(
                max_deg, 
                model,
                lonlatheight,
                model_dir,
                chunk_size,
                parallel,
                ellipsoid,
                output_dir,
                tide_system,
                converted,
                input_file,
                bbox,
                bbox_offset,
                grid_size,
                grid_unit
            )
        elif task == 'reference-geoid':
            results['reference-geoid'] = compute_geoid(
                max_deg, 
                model,
                model_dir,
                chunk_size,
                parallel,
                ellipsoid,
                output_dir,
                tide_system,
                converted,
                input_file,
                bbox,
                bbox_offset,
                grid_size,
                grid_unit
            )
    
    output_files = [result['output_file'] for result in results.values()]
    
    return {'status': 'success', 'output_files': output_files}


def main() -> None:
    '''
    Main function for synthesizing gravity functionals from a global geopotential model. The functionals include the 
    gravity anomalies (Dg_ggm) and gridded reference geoid (N_ref).
    Supports downloading the GGM, computing gravity anomalies (Dg_ggm), and/or gridded reference geoid (N_res)
    '''
    
    parser = argparse.ArgumentParser(
        description=(
            'Calculate reference gravity anomalies and geoid heights from a Global Geopotential Model (GGM). '
            'Supports downloading GGM files from ICGEM, computing gravity anomalies at the observation points, and '
            'calculating gridded reference geoid heights. Use --do to select specific tasks'
        )
    )
    parser.add_argument('--model', type=str, required=True, 
                        help='GGM name with (e.g., EGM2008.gfc) or without .gfc extension (e.g., EGM2008)')
    parser.add_argument('--model-dir', type=str, default=None, 
                        help='Directory where the GGM is located. Defaults to <proj-name>/downloads if not specified')
    parser.add_argument('--do', type=str, default='all', choices=['download', 'gravity-anomaly', 'reference-geoid', 'all'], 
                        help='Computation steps to perform: [download, gravity-anomaly, reference-geoid, or all (default: all)]')
    parser.add_argument('--start', type=str, default=None, 
                        help='Start processing from this step: [download, gravity-anomaly, reference-geoid]')
    parser.add_argument('--end', type=str, default=None, 
                        help='End processing at this step: [download, gravity-anomaly, reference-geoid]')
    parser.add_argument('--max-deg', type=int, default=90, 
                        help='Maximum degree of truncation for the computations')
    parser.add_argument('--grid-size', type=float, default=None, 
                        help='Grid size/resolution in degrees, minutes, or seconds. Required if --grid')
    parser.add_argument('--grid-unit', type=str, default=None, choices=['degrees', 'minutes', 'seconds'], 
                        help='Unit of the grid size. Required if --grid')
    parser.add_argument('--bbox', type=float, nargs=4, default=[None, None, None, None], 
                        help='The bounding box [W,E,S,N] of the study area. Required if --grid')
    parser.add_argument('--bbox-offset', type=float, default=1.0, 
                        help='Offset around the bounding box over which to grid the gravity anomalies. Required if --grid')
    parser.add_argument('--input-file', type=str, 
                        help='Input file with lon, lat, and height data')
    parser.add_argument('--chunk-size', type=int, default=500, 
                        help='Chunk size for parallel processing')
    parser.add_argument('--parallel', action='store_true', default=False, 
                        help='Enable parallel processing')
    parser.add_argument('--ellipsoid', type=str, default='wgs84', 
                        help='Reference ellipsoid. Supported: [wgs84, grs80]')
    parser.add_argument('--proj-name', type=str, default='GeoidProject', 
                        help='Project directory where downloads and results subdirectories are created')
    parser.add_argument('--gravity-tide', type=str, default=None, 
                        help='Tide system of the surface gravity data (required for gravity-anomaly): Options: [mean_tide, zero_tide, tide_free]')
    parser.add_argument('--converted', action='store_true', default=False,
                        help='Indicate that input data is already in the target tide system')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir) if args.model_dir is not None else Path(args.proj_name) / 'downloads'
    output_dir = Path(args.proj_name) / 'results'
    
    # Set up directory structure
    directory_setup(args.proj_name)
    
    # Load lonlatheight data if needed
    lonlatheight = None
    if args.input_file and (args.do in ['gravity-anomaly', 'all']):
        input_path = Path(args.input_file)
        if input_path.suffix == '.csv':
            lonlatheight = pd.read_csv(input_path)
        elif input_path.suffix in ['.xlsx', '.xls']:
            lonlatheight = pd.read_excel(input_path)
        elif input_path.suffix == '.txt':
            lonlatheight = pd.read_csv(input_path, delimiter='\t')
        elif input_path.suffix in ['.npy', '.npz']:
            lonlatheight = np.load(input_path)
            lonlatheight = pd.DataFrame(lonlatheight, columns=['lon', 'lat', 'height'])
        else:
            raise ValueError(f'Unsupported file format: {input_path.suffix}')
        
    # Validate parameters
    validate_params(args, lonlatheight)
    
    # Run computation
    result = compute(
        max_deg=args.max_deg,
        model=args.model,
        lonlatheight=lonlatheight,
        model_dir=model_dir,
        chunk_size=args.chunk_size,
        parallel=args.parallel,
        ellipsoid=args.ellipsoid,
        output_dir=output_dir,
        do=args.do,
        start=args.start,
        end=args.end,
        tide_system=args.gravity_tide,
        converted=args.converted,
        input_file=args.input_file,
        bbox=args.bbox,
        bbox_offset=args.bbox_offset,
        grid_size=args.grid_size,
        grid_unit=args.grid_unit
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())