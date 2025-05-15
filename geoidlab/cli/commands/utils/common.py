############################################################
# Geoid workflow CLI interface                             #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import numpy as np

from pathlib import Path



def validate_params(args, lonlatheight=None) -> None:
    '''
    Validate parameters for GGM reference computations.
    
    Parameters
    ----------
    args        : Parsed argparse arguments
    lonlatheight: DataFrame with columns lon, lat, and height (optional)
    
    Raises
    ------
    ValueError  : If parameters are invalid
    '''
    # Validate max-deg from computation tasks
    if args.do in ['gravity-anomaly', 'reference-geoid', 'all']:
        if args.max_deg <= 0:
            raise ValueError('max_deg must be greater than 0')
        
    # Validate chunk-size for computation tasks
    if args.do in ['gravity-anomaly', 'reference-geoid', 'all']:
        if args.chunk_size <= 0 and args.parallel:
            raise ValueError('chunk_size must be greater than 0')

    # Validate ellipsoid
    if args.ellipsoid not in ['wgs84', 'grs80']:
        raise ValueError('ellipsoid must be wgs84 or grs80')

    # Validate model
    if not args.model:
        raise ValueError('model is required')



def directory_setup(project_dir: str = None) -> None:
    '''
    Create project directory
    '''
    
    # Set up project directory
    project_dir = Path(project_dir).resolve() if project_dir else Path.cwd() / 'GeoidProject'
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Sub-directories
    downloads_dir = project_dir / 'downloads'
    results_dir = project_dir / 'results'
    figures_dir = results_dir / 'figures'
    
    # Create sub-directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    
def get_grid_lon_lat(grid_extent, grid_resolution, unit) -> tuple[np.ndarray, np.ndarray]:
    '''
    Given grid extent and grid resolution, return grid latitudes and longitudes.
    
    Parameters
    ----------
    grid_extent     : the extent of the grid (lon_min, lon_max, lat_min, lat_max) (W,E,S,N)
    grid_resolution : the resolution of the grid (degrees, minutes, or seconds)
    unit            : the unit of grid_resolution
    
    Returns
    -------
    lon_grid        : array of longitudes
    lat_grid        : array of latitudes
    '''
    
    # Convert resolution to degrees if in minutes or seconds
    if unit == 'minutes':
        grid_resolution = grid_resolution / 60.0
    elif unit == 'seconds':
        grid_resolution = grid_resolution / 3600.0
    else:
        grid_resolution = grid_resolution
    
    # Create the grid 
    lon_min, lon_max, lat_min, lat_max = grid_extent
    num_x_points = int((lon_max - lon_min) / grid_resolution) + 1
    num_y_points = int((lat_max - lat_min) / grid_resolution) + 1
    
    lon_grid = np.linspace(lon_min, lon_max, num_x_points)
    lat_grid = np.linspace(lat_min, lat_max, num_y_points)
    
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
    return lon_grid, lat_grid

def to_seconds(resolution, unit) -> float:
    '''
    Convert resolution to seconds
    '''
    if unit == 'minutes':
        resolution *= 60.0
    elif unit == 'degrees':
        resolution *= 3600.0
    else:
        resolution = resolution
        
    return resolution