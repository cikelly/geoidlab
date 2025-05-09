############################################################
# Geoid workflow CLI interface                             #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

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
    downloads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)