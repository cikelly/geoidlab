############################################################
# GGM synthesize CLI interface                             #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys

import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path

from geoidlab.cli.commands.utils.common import(
    directory_setup,
    get_grid_lon_lat
)
from geoidlab.icgem import download_ggm, get_ggm_tide_system
from geoidlab.utils.io import save_to_netcdf
from geoidlab.tide import GravityTideSystemConverter
from geoidlab.ggm import GlobalGeopotentialModel

class GGMSynthesis():
    '''
    Class to organize reference.py CLI into a modular easily maintainable CLI tool
    '''
    # Map CLI tasks to methods and output configurations
    TASK_CONFIG = {
        'download': {'method': 'download', 'output': None},
        'gravity-anomaly': {
            'method': 'compute_gravity_anomaly',
            'ggm_method': 'gravity_anomaly',
            'output': {'key': 'Dg_ggm', 'file': 'Dg_ggm'}
        },
        'reference-geoid': {
            'method': 'compute_geoid',
            'ggm_method': 'geoid',
            'output': {'key': 'N_ref', 'file': 'N_ref'}
        },
        'height-anomaly': {
            'method': 'compute_height_anomaly',
            'ggm_method': 'height_anomaly',
            'output': {'key': 'zeta', 'file': 'zeta'}
        },
        'gravity-disturbance': {
            'method': 'compute_gravity_disturbance',
            'ggm_method': 'gravity_disturbance',
            'output': {'key': 'dg', 'file': 'dg'}
        },
        # Add more tasks (e.g., disturbing_potential, second_radial_derivative)
    }

    def __init__(
        self,
        model: str,
        max_deg: int = 90,
        model_dir: str | Path = None,
        output_dir: str | Path = 'results',
        ellipsoid: str = 'wgs84',
        chunk_size: int = 500,
        parallel: bool = False,
        tide_system: str = None,
        converted: bool = False,
        input_file: str = None,
        bbox: list = None,
        bbox_offset: float = 1.0,
        grid_size: float = None,
        grid_unit: str = 'minutes',
        proj_name: str = 'GeoidProject',
        icgem: bool = False
    ) -> None:
        '''
        Initialize GGMSynthesis class
        
        Parameters
        ----------
        icgem     : Whether to use ICGEM's version of N_ggm
        '''
        self.model = model
        self.max_deg = max_deg
        self.model_dir = Path(model_dir) if model_dir else Path(proj_name) / 'downloads'
        self.output_dir = Path(output_dir)
        self.ellipsoid = ellipsoid
        self.chunk_size = chunk_size
        self.parallel = parallel
        self.tide_system = tide_system
        self.converted = converted
        self.input_file = input_file
        self.bbox = bbox
        self.bbox_offset = bbox_offset
        self.grid_size = grid_size
        self.grid_unit = grid_unit
        self.proj_name = proj_name
        self.lonlatheight = None
        self.lon_grid = None
        self.lat_grid = None
        self.icgem = icgem
        
        # Directory setup
        directory_setup(proj_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._validate_params()
        
    def _validate_params(self) -> None:
        '''Validate parameters'''
        if self.ellipsoid not in ['wgs84', 'grs80']:
            raise ValueError('Ellipsoid must be \'wgs84\' or \'grs80\'')
        if self.tide_system and self.tide_system not in ['mean_tide', 'zero_tide', 'tide_free']:
            raise ValueError('Tide system must be one of: mean_tide, zero_tide, tide_free')
        if self.bbox and len(self.bbox) == 4:
            min_lon, max_lon, min_lat, max_lat = self.bbox
            if not (min_lon <= max_lon and min_lat <= max_lat):
                raise ValueError('Invalid bbox: west must be <= east, south <= north')
            
    def _process_input(self) -> None:
        '''Load input file or generate grid.'''
        if self.input_file:
            input_path = Path(self.input_file)
            if input_path.suffix == '.csv':
                self.lonlatheight = pd.read_csv(input_path)
            elif input_path.suffix in ['.xlsx', '.xls']:
                self.lonlatheight = pd.read_excel(input_path)
            elif input_path.suffix == '.txt':
                self.lonlatheight = pd.read_csv(input_path, delimiter='\t')
            elif input_path.suffix in ['.npy', '.npz']:
                self.lonlatheight = pd.DataFrame(np.load(input_path), columns=['lon', 'lat', 'height'])
            else:
                raise ValueError(f'Unsupported file format: {input_path.suffix}')
        elif self.bbox and self.grid_size and self.grid_unit:
            min_lon, max_lon, min_lat, max_lat = self.bbox
            min_lon -= self.bbox_offset
            max_lon += self.bbox_offset
            min_lat -= self.bbox_offset
            max_lat += self.bbox_offset
            grid_extent = (min_lon, max_lon, min_lat, max_lat)
            self.lon_grid, self.lat_grid = get_grid_lon_lat(grid_extent, self.grid_size, self.grid_unit)
            self.lonlatheight = pd.DataFrame({
                'lon': self.lon_grid.flatten(),
                'lat': self.lat_grid.flatten(),
                'height': 0,
                'gravity': 0
            })
        else:
            raise ValueError('Either input_file or bbox, grid_size, grid_unit must be provided')
        
    def _convert_tide_system(self, model_path: Path) -> Path | None:
        '''Convert input data to GGM tide system.'''
        if not self.converted and self.tide_system:
            ggm_tide = get_ggm_tide_system(icgem_file=model_path, model_dir=self.model_dir)
            if ggm_tide != self.tide_system:
                print(f'Converting input data from {self.tide_system} to {ggm_tide} system...')
                converter = GravityTideSystemConverter(data=self.lonlatheight)
                input_filename = Path(self.input_file).stem if self.input_file else 'lonlatheight'
                converted_data_path = self.output_dir / f'{input_filename}_{ggm_tide}.csv'
                self.lonlatheight.to_csv(converted_data_path, index=False)
                print(f'Converted data saved to {converted_data_path}')
                return converted_data_path
        return None
    
    def download(self) -> dict:
        '''Step to download GGM from ICGEM'''
        model_path = (self.model_dir / self.model).with_suffix('.gfc')
        if not model_path.exists():
            download_ggm(self.model, self.model_dir)
        return {'status': 'success', 'output_file': str(model_path)}
    
    def _compute_functional(self, ggm_method: str, task_name: str, output_key: str, output_file: str, icgem: bool = None) -> dict:
        '''Compute a gravity functional using GlobalGeopotentialModel.'''
        print(f'\nComputing {task_name} with max_deg={self.max_deg}, ellipsoid={self.ellipsoid}')
        model_path = (self.model_dir / self.model).with_suffix('.gfc')
        output_file = self.output_dir / f'{output_file}.{'csv' if self.input_file else 'nc'}'

        # Process input if not already done
        if self.lonlatheight is None:
            self._process_input()

        # Handle tide conversion
        # if self.input_file:
        converted_data_path = self._convert_tide_system(model_path)

        # Initialize model
        model = GlobalGeopotentialModel(
            model_name=model_path.stem,
            grav_data=self.lonlatheight,
            nmax=self.max_deg,
            zonal_harmonics=True,
            ellipsoid=self.ellipsoid,
            model_dir=self.model_dir,
            chunk_size=self.chunk_size
        )

        # Compute functional, use if to pass method specific parameters
        if ggm_method == 'geoid' and icgem is not None:
            result = getattr(model, ggm_method)(parallel=self.parallel, icgem=icgem)
        else:
            result = getattr(model, ggm_method)(parallel=self.parallel)

        # Save output
        if self.input_file:
            df = pd.DataFrame({
                'lon': self.lonlatheight['lon'],
                'lat': self.lonlatheight['lat'],
                output_key: result
            })
            df.to_csv(output_file, index=False)
        else:
            result = result.reshape(self.lon_grid.shape)
            save_to_netcdf(
                data=result,
                lon=self.lon_grid,
                lat=self.lat_grid,
                dataset_key=output_key,
                filepath=output_file
            )

        print(f'{task_name} written to {output_file}\n')
        return {
            'status': 'success',
            'output_file': str(output_file),
            'converted_data': str(converted_data_path) if converted_data_path else None
        }

    def compute_gravity_anomaly(self) -> dict:
        return self._compute_functional(
            ggm_method='gravity_anomaly',
            task_name='Gravity anomalies',
            output_key='Dg_ggm',
            output_file='Dg_ggm'
        )

    def compute_geoid(self) -> dict:
        return self._compute_functional(
            ggm_method='geoid',
            task_name='Reference geoid heights',
            output_key='N_ref',
            output_file='N_ref',
            icgem=self.icgem
        )

    def compute_height_anomaly(self) -> dict:
        return self._compute_functional(
            ggm_method='height_anomaly',
            task_name='Height anomalies',
            output_key='zeta',
            output_file='zeta'
        )

    def compute_gravity_disturbance(self) -> dict:
        return self._compute_functional(
            ggm_method='gravity_disturbance',
            task_name='Gravity disturbances',
            output_key='dg',
            output_file='dg'
        )

    def run(self, tasks: list) -> dict:
        results = {}
        for task in tasks:
            if task not in self.TASK_CONFIG:
                raise ValueError(f'Unknown task: {task}')
            config = self.TASK_CONFIG[task]
            method = getattr(self, config['method'])
            results[task] = method()
        output_files = [result['output_file'] for result in results.values() if result.get('output_file')]
        return {'status': 'success', 'output_files': output_files}

def main() -> 0:
    parser = argparse.ArgumentParser(
        description=(
            'Calculate gravity functionals and geoid heights from a Global Geopotential Model (GGM). '
            'Supported tasks: download, gravity-anomaly, reference-geoid, height-anomaly, gravity-disturbance.'
        )
    )
    parser.add_argument('--model', type=str, required=True, 
                        help='GGM name (e.g., EGM2008)')
    parser.add_argument('--model-dir', type=str, default=None, 
                        help='Directory for GGM files')
    parser.add_argument('--do', type=str, default='all', 
                        choices=['download', 'gravity-anomaly', 'reference-geoid', 'height-anomaly', 'gravity-disturbance', 'all'],
                        help='Computation steps to perform')
    parser.add_argument('--start', type=str, default=None, 
                        help='Start task')
    parser.add_argument('--end', type=str, default=None, 
                        help='End task')
    parser.add_argument('--max-deg', type=int, default=90, 
                        help='Maximum degree of truncation')
    parser.add_argument('--grid-size', type=float, default=None, 
                        help='Grid size in degrees, minutes, or seconds')
    parser.add_argument('--grid-unit', type=str, default=None, 
                        choices=['degrees', 'minutes', 'seconds'],
                        help='Unit of grid size')
    parser.add_argument('--bbox', type=float, nargs=4, default=[None, None, None, None], 
                        help='Bounding box [W,E,S,N] in degrees')
    parser.add_argument('--bbox-offset', type=float, default=1.0, 
                        help='Offset around bounding box')
    parser.add_argument('--input-file', type=str, 
                        help='Input file with lon, lat, height')
    parser.add_argument('--chunk-size', type=int, default=500, 
                        help='Chunk size for parallel processing')
    parser.add_argument('--parallel', action='store_true', default=False, 
                        help='Enable parallel processing')
    parser.add_argument('--ellipsoid', type=str, default='wgs84', 
                        help='Reference ellipsoid: wgs84, grs80')
    parser.add_argument('--proj-name', type=str, default='GeoidProject', 
                        help='Project directory')
    parser.add_argument('--gravity-tide', type=str, default=None, 
                        help='Tide system: mean_tide, zero_tide, tide_free')
    parser.add_argument('--converted', action='store_true', default=False,
                        help='Input data is in target tide system')
    parser.add_argument('--icgem', action='store_true', default=False,
                        help='Use ICGEM formula for reference geoid computation (only for reference-geoid task)')

    args = parser.parse_args()

    workflow = ['download', 'gravity-anomaly', 'reference-geoid', 'height-anomaly', 'gravity-disturbance']
    if args.do != 'all' and (args.start or args.end):
        raise ValueError('Cannot specify both --do and --start/--end')
    if args.do == 'all':
        tasks = workflow
    elif args.start or args.end:
        start_idx = 0 if args.start is None else workflow.index(args.start)
        end_idx = len(workflow) - 1 if args.end is None else workflow.index(args.end)
        tasks = workflow[start_idx:end_idx + 1]
    else:
        tasks = [args.do]

    workflow = GGMSynthesis(
        model=args.model,
        max_deg=args.max_deg,
        model_dir=args.model_dir,
        output_dir=Path(args.proj_name) / 'results',
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        parallel=args.parallel,
        tide_system=args.gravity_tide,
        converted=args.converted,
        input_file=args.input_file,
        bbox=args.bbox,
        bbox_offset=args.bbox_offset,
        grid_size=args.grid_size,
        grid_unit=args.grid_unit,
        proj_name=args.proj_name,
        icgem=args.icgem
    )

    result = workflow.run(tasks)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())