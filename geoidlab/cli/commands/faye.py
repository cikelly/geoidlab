############################################################
# Gravity Reduction CLI interface                          #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import pandas as pd
import numpy as np

from pathlib import Path

from geoidlab.cli.commands.utils.common import directory_setup
from geoidlab.utils.interpolators import Interpolators
from geoidlab import gravity
from geoidlab.tide import GravityTideSystemConverter
from geoidlab.icgem import get_ggm_tide_system
from geoidlab.cli.commands.utils.common import directory_setup, get_grid_lon_lat

class GravityReduction:
    '''Class to perform gravity reductions (Free-air, Bouguer, Faye/Helmert anomalies)'''
    TASK_CONFIG = {
        'free-air': {
            'method': 'compute_anomalies',
            'output': {'key': 'free_air', 'file': 'free_air'},
            'anomaly_type': 'free_air'
        },
        'bouguer': {
            'method': 'compute_anomalies',
            'output': {'key': 'bouguer', 'file': 'bouguer'},
            'anomaly_type': 'bouguer'
        },
        'faye': {
            'method': 'compute_faye',
            'output': {'key': 'faye', 'file': 'faye'}
        }
    }
    
    def __init__(
        self,
        input_file: str,
        marine_data: str = None,
        gravity_tide: str = None,
        ellipsoid: str = 'wgs84',
        converted: bool = False,
        grid: bool = False,
        grid_size: float = None,
        bbox: list = [None, None, None, None],
        bbox_offset: float = 1.,
        proj_name: str = 'GeoidProject',
    ) -> None:
        self.input_file = input_file
        self.marine_data = marine_data
        self.gravity_tide = gravity_tide
        self.ellipsoid = ellipsoid
        self.converted = converted
        self.grid = grid
        self.grid_size = grid_size
        self.bbox = bbox
        self.bbox_offset = bbox_offset
        self.proj_name = proj_name
        self.output_dir = Path(proj_name) / 'results'
        self.lonlatheight = None
        self.lon_grid = None
        self.lat_grid = None
        self.free_air = None

        directory_setup(proj_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_params()

        def _validate_params(self) -> None:
            if not Path(self.input_file).is_file():
                raise ValueError(f'Input file {self.input_file} does not exist')
            if self.marine_data and not Path(self.marine_data).is_file():
                raise ValueError(f'Marine data file {self.marine_data} does not exist')
            if self.ellipsoid not in ['wgs84', 'grs80']:
                raise ValueError('Ellipsoid must be wgs84 or grs80')
            if self.gravity_tide and self.gravity_tide not in ['mean_tide', 'zero_tide', 'tide_free']:
                raise ValueError('Gravity tide must be mean_tide, zero_tide, or tide_free')
            if self.grid:
                if not (self.grid_size and self.bbox):
                    raise ValueError('grid-size and bbox are required when --grid is used')
                min_lon, max_lon, min_lat, max_lat = self.bbox
                if not (min_lon <= max_lon and min_lat <= max_lat):
                    raise ValueError('Invalid bbox: west must be <= east, south <= north')

        def _process_input(self) -> None:
            '''Load input file and marine data.'''
            input_path = Path(self.input_file)
            if input_path.suffix == '.csv':
                self.lonlatheight = pd.read_csv(input_path)
            elif input_path.suffix in ['.xlsx', '.xls']:
                self.lonlatheight = pd.read_excel(input_path)
            elif input_path.suffix == '.txt':
                self.lonlatheight = pd.read_csv(input_path, delimiter='\t')
            else:
                raise ValueError(f'Unsupported file format: {input_path.suffix}')
            if self.marine_data:
                marine_path = Path(self.marine_data)
                if marine_path.suffix == '.csv':
                    marine_data = pd.read_csv(marine_path)
                elif marine_path.suffix in ['.xlsx', '.xls']:
                    marine_data = pd.read_excel(marine_path)
                elif marine_path.suffix == '.txt':
                    marine_data = pd.read_csv(marine_path, delimiter='\t')
                else:
                    raise ValueError(f'Unsupported file format: {marine_path.suffix}')
                self.lonlatheight = pd.concat([self.lonlatheight, marine_data], ignore_index=True)
                
        def _process_grid(self) -> None:
            '''Generate grid for output if --grid is specified.'''
            if self.grid and self.bbox and self.grid_size:
                min_lon, max_lon, min_lat, max_lat = self.bbox
                offset = self.bbox_offset
                grid_extent = (min_lon - offset, max_lon + offset, min_lat - offset, max_lat + offset)
                self.lon_grid, self.lat_grid = get_grid_lon_lat(grid_extent, self.grid_size, 'minutes')
                self.lonlatheight = pd.DataFrame({
                    'lon': self.lon_grid.flatten(),
                    'lat': self.lat_grid.flatten(),
                    'height': 0,
                    'gravity': 0
                })
        
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
                else:
                    print('Surface gravity and GGM have the same tide system. Skipping conversion.')
            return None
        
        def compute_anomalies(self, anomaly_type='free_air') -> None:
            '''Compute Free-air and Bouguer anomalies, return requested type.'''
            if self.lonlatheight is None:
                self._process_input()
            if self.grid:
                self._process_grid()
            converted_path = self._convert_tide_system()
            free_air = gravity.free_air_anomaly(
                lon=self.lonlatheight['lon'], lat=self.lonlatheight['lat'],
                height=self.lonlatheight['height'], gravity=self.lonlatheight['gravity'],
                ellipsoid=self.ellipsoid
            )
            bouguer = gravity.bouguer_anomaly(
                lon=self.lonlatheight['lon'], lat=self.lonlatheight['lat'],
                height=self.lonlatheight['height'], gravity=self.lonlatheight['gravity'],
                ellipsoid=self.ellipsoid
            )
            self.free_air = free_air  # Cache for Faye computation
            output_file = self.output_dir / f'{anomaly_type}.csv'
            result = free_air if anomaly_type == 'free_air' else bouguer
            df = pd.DataFrame({
                'lon': self.lonlatheight['lon'],
                'lat': self.lonlatheight['lat'],
                anomaly_type: result
            })
            df.to_csv(output_file, index=False)
            print(f'{anomaly_type.replace("_", " ").title()} anomalies written to {output_file}')
            return {
                'status': 'success',
                'output_file': str(output_file),
                'converted_data': str(converted_path) if converted_path else None
            }
            
        def compute_faye(self) -> dict:
            '''Compute Faye/Helmert anomalies using Free-air and terrain corrections.'''
            if self.lonlatheight is None:
                self._process_input()
            if self.grid:
                self._process_grid()
            if self.free_air is None:
                self.compute_anomalies('free_air')  # Ensure Free-air is computed
            # Placeholder for TC computation at points (new method from terrain.py)
            # tc = TerrainQuantities.compute_tc_points(...)  # To be implemented
            # faye = self.free_air + tc
            output_file = self.output_dir / 'faye.csv'
            # df = pd.DataFrame({
            #     'lon': self.lonlatheight['lon'],
            #     'lat': self.lonlatheight['lat'],
            #     'faye': faye
            # })
            # df.to_csv(output_file, index=False)
            # print(f'Faye anomalies written to {output_file}')
            return {
                'status': 'success',
                'output_file': str(output_file),
                'converted_data': None
            }
            
        def run(self, tasks) -> None:
            '''Execute specified tasks in order.'''
            results = {}
            for task in tasks:
                if task not in self.TASK_CONFIG:
                    raise ValueError(f'Unknown task: {task}')
                config = self.TASK_CONFIG[task]
                method = getattr(self, config['method'])
                if 'anomaly_type' in config:
                    results[task] = method(anomaly_type=config['anomaly_type'])
                else:
                    results[task] = method()
            output_files = [result['output_file'] for result in results.values() if result.get('output_file')]
            return {'status': 'success', 'output_files': output_files}


def main() -> None:
    '''
    Main function for gravity reductions. The supported methods are Free-air and Bouguer reductions,
    with outputs including Free-air and Bouguer anomalies, and Helmert/Faye anomalies
    '''
    
    parser = argparse.ArgumentParser(
        description=(
            'Perform gravity reduction'
        )
    )
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Input file with lon, lat, gravity, and height data (required)')
    parser.add_argument('-m', '--marine-data', type=str, default=None,
                        help='Input file with lon, lat, gravity, and height data for marine gravity anomalies (optional)')
    parser.add_argument('--do', type=str, default='free-air', choices=['free-air', 'bouguer', 'faye', 'all'],
                        help='Computation steps to perform: [free-air, bouguer, faye, or all (default: free-air)]')
    parser.add_argument('-s', '--start', type=str, default=None,
                        help='Start processing from this step: [free-air, bouguer, faye]')
    parser.add_argument('-e', '--end', type=str, default=None,
                        help='End processing at this step: [free-air, bouguer, faye]')
    parser.add_argument('-gt', '--gravity-tide', type=str, default=None, 
                        help='Tide system of the surface gravity data (required for gravity-anomaly): Options: [mean_tide, zero_tide, tide_free]')
    parser.add_argument('-g', '--grid', action='store_true', default=False,
                        help='Grid the gravity anomalies over a bounding box')
    parser.add_argument('-gs', '--grid-size', type=float, default=None, 
                        help='Grid size in minutes (e.g., 5 for a 5-by-5 minute grid). Required if --grid')
    parser.add_argument('-gu', '--grid-unit', type=str, default=None, 
                        choices=['degrees', 'minutes', 'seconds'],
                        help='Unit of grid size')
    parser.add_argument('-b', '--bbox', type=list, default=[None, None, None, None], 
                        help='The bounding box [W,S,E,N] of the study area. Required if --grid')
    parser.add_argument('-bo', '--bbox-offset', type=float, default=1.0, 
                        help='Offset around the bounding box [W,S,E,N] over which to grid the gravity anomalies. Required if --grid')
    parser.add_argument('-ell', '--ellipsoid', type=str, default='wgs84', 
                        help='Reference ellipsoid. Supported: [wgs84, grs80]')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject', 
                        help='Project directory where downloads and results subdirectories are created')
    parser.add_argument('-c', '--converted', action='store_true', default=False,
                        help='Indicate that input data is already in the target tide system')
    parser.add_argument('--interp-method', type=str, default='kriging', choices=['linear', 'spline', 'kriging', 'rbf', 'idw', 'biharmonic', 'gpr', 'lsc'],
                        help='Interpolation method for gridding anomalies')
    parser.add_argument('--int-merge', action='store_true', default=False,
                        help=('Flag to combine interpolation methods with linear extrapolation. Necessary because some of the methods do not extrapolate (well)'
                              'outside the convex hull. This is particularly useful for geoid computation, where you may not have data covering --bbox-offset. (Default: False)'))
    
    
    
    args = parser.parse_args()
    workflow = ['free-air', 'bouguer', 'faye']
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

    reduction = GravityReduction(
        input_file=args.input_file, marine_data=args.marine_data,
        gravity_tide=args.gravity_tide, ellipsoid=args.ellipsoid,
        converted=args.converted, grid=args.grid, grid_size=args.grid_size,
        bbox=args.bbox, bbox_offset=args.bbox_offset, proj_name=args.proj_name
    )
    result = reduction.run(tasks)
    return 0

if __name__ == '__main__':
    sys.exit(main())