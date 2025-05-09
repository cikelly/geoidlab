############################################################
# Geoid workflow CLI interface                             #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import pandas as pd
import numpy as np

from pathlib import Path

from geoidlab.cli.commands.utils.common import validate_params, directory_setup
from geoidlab.icgem import download_ggm, read_icgem

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
    else:
        print(f'{model} already exists in {model_dir}')

    return {'status': 'success', 'output_file': str(model_path)}

def compute_gravity_anomaly(
    max_deg: int, 
    model: str,
    lonlatheight: pd.DataFrame | np.ndarray,
    model_dir: str | Path = None,
    chunk_size: int = 500,
    parallel: bool = False,
    ellipsoid: str = 'wgs84',
    output_dir: str | Path = 'results'
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
    
    Returns
    -------
    dict        : Dictionary with status and output file path
    '''
    output_file = Path(output_dir) / 'Dg_ggm.csv'
    print(f'\nGravity anomalies will be computed with max_deg={max_deg}, ellipsoid={ellipsoid}')
    
    model_dir = Path('.') if model_dir is None else Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = (model_dir / model).with_suffix('.gfc')
    
    print(f'\nReading {model_path.stem} file...')
    shc = read_icgem(model_path, model_dir)
    
    return {'status': 'success', 'output_file': str(output_file)}
    

def compute_geoid(
    max_deg, 
    model,
    model_dir: str | Path = 'downloads',
    chunk_size: int = 500,
    parallel: bool = True,
    ellipsoid: str = 'wgs84',
    output_dir: str | Path = 'results'
) -> dict:
    '''
    Compute reference geoid heights (N_ref) on the final grid
    
    Parameters
    ----------
    max_deg   : Maximum degree of truncation for the computations
    model     : Name of the GGM (with or without .gfc extension)
    model_dir : Directory to download GGM to
    chunk_size: Chunk size for parallel processing
    parallel  : Enable parallel processing
    ellipsoid : Reference ellipsoid
    output_dir: Directory to write outputs to
    
    Returns
    -------
    dict      : Dictionary with status and output file path
    '''
    print(f'Computing reference anomalies and geoid heights with max_deg={max_deg}, ellipsoid={ellipsoid}')
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
    do: str = 'all'
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
    
    Returns
    -------
    dict        : Dictionary with status and output file(s)
    '''
    results = {}
    
    if do == 'download' or do == 'all':
        results['download'] = download(model, model_dir)
    
    if do == 'gravity-anomaly' or do == 'all':
        if lonlatheight is None:
            raise ValueError('lonlatheight data is required for gravity-anomaly task')
        results['gravity-anomaly'] = compute_gravity_anomaly(
            max_deg, 
            model,
            lonlatheight,
            model_dir,
            chunk_size,
            parallel,
            ellipsoid,
            output_dir
        )
    
    if do == 'reference-geoid' or do == 'all':
        results['reference-geoid'] = compute_geoid(
            max_deg, 
            model,
            model_dir,
            chunk_size,
            parallel,
            ellipsoid,
            output_dir
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
    parser.add_argument('--do', type=str, default='all', choices=['download', 'gravity-anomaly', 'reference-geoid', 'all'], 
                        help='Computation steps to perform: [download, gravity-anomaly, reference-geoid, or all (default: all)]')
    parser.add_argument('--max-deg', type=int, default=90, help='Maximum degree of truncation for the computations')
    parser.add_argument('--model', type=str, required=True, help='GGM name with (e.g., EGM2008.gfc) or without .gfc extension (e.g., EGM2008)')
    parser.add_argument('--input-file', type=str, help='Input file with lon, lat, and height data (required for gravity-anomaly)')
    parser.add_argument('--model-dir', type=str, default=None, help='Directory where the GGM is located. Defaults to <proj-name>/downloads if not specified')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size for parallel processing')
    parser.add_argument('--parallel', action='store_true', default=False, help='Enable parallel processing')
    parser.add_argument('--ellipsoid', type=str, default='wgs84', help='Reference ellipsoid. Supported: [wgs84, grs80]')
    parser.add_argument('--proj-name', type=str, default='GeoidProject', help='Project directory where downloads and results subdirectories are created')
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
        do=args.do
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())