############################################################
# Topographic quantities CLI interface                     #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path

from geoidlab.cli.commands.utils.common import directory_setup, to_seconds
from geoidlab import terrain
from geoidlab.dem import dem4geoid

class TopographicQuantities:
    '''
    CLI to compute topographic quantities from a Digital Elevation Model (DEM) and/or a reference DEM
    Supported tasks: download, terrain-correction, rtm-anomaly, indirect-effect, height-anomaly, site
    '''
    TASK_CONFIG = {
        'download': {'method': 'download', 'output': None},
        'terrain-correction': {
            'method': 'compute_tc', 
            'terrain_method': 'terrain_correction',
            'output': {'key': 'tc', 'file': 'TC'}
        },
        'rtm-anomaly': {
            'method': 'compute_rtm', 
            'terrain_method': 'rtm_anomaly',
            'output': {'key': 'dg_RTM', 'file': 'dg_RTM'},
        },
        'indirect-effect': {
            'method': 'compute_ind',
            'terrain_method': 'indirect_effect',
            'output': {'key': 'ind', 'file': 'N_ind'}
        },
        'height-anomaly': {
            'method': 'compute_rtm_height',
            'terrain_method': 'rtm_height_anomaly',
            'output': {'key': 'zeta_rtm', 'file': 'zeta_rtm'}
        },
        'site': {
            'method': 'compute_site',
            'terrain_method': 'site',
            'output': {'key': 'Dg_site', 'file': 'Dg_SITE'}
        }
    }
    
    def __init__(
        self,
        topo: str,
        ref_topo: str = None,
        model_dir: str | Path = None,
        output_dir: str | Path = 'results',
        ellipsoid: str = 'wgs84',
        chunk_size: int = 500,
        radius: float = 110.,
        proj_name: str = 'GeoidProject',
        bbox: list[float] = None,
        bbox_offset: float = 1.0,
        grid_size: float = None,
        grid_unit: str = 'seconds',
        window_mode: str = 'radius',
        parallel: bool = False,
        # resolution: int = 30,
        # resolution_unit: str = 'seconds',
        interp_method: str = 'slinear',
        approximation: bool = False,
        tc: xr.Dataset = None,
    ) -> None:
        '''
        
        Parameters
        ----------
        topo           : Path to DEM file
        ref_topo       : Path to reference topo file
        model_dir      : Directory for DEM files
        output_dir     : Directory for output files
        ellipsoid      : Ellipsoid to use
        chunk_size     : Chunk size for parallel processing
        radius         : Search radius in kilometers
        proj_name      : Name of the project
        bbox           : Bounding box [W, E, S, N]
        bbox_offset    : Offset around bounding box
        grid_size      : Grid size in degrees, minutes, or seconds
        grid_unit      : Unit of grid size
        window_mode    : Method for selecting sub-grid for computation. Options: 'radius', 'fixed'
        parallel       : Use parallel processing
        resolution     : Target resolution of the DEM in arc seconds
        resolution_unit: Unit of target resolution
        interp_method  : Interpolation method for resampling DEM ('linear', 'slinear', 'cubic', 'quintic')
        approximation  : Use the approximate formula for RTM gravity anomalies
        tc             : Terrain correction. Necessary to avoid recomputing TC for RTM anomalies
        
        Returns
        -------
        None
        '''
        self.topo = topo
        self.ref_topo = ref_topo
        self.model_dir = Path(model_dir) if model_dir else Path(proj_name) / 'downloads'
        self.output_dir = Path(output_dir)
        self.ellipsoid = ellipsoid
        self.chunk_size = chunk_size
        self.radius = radius
        self.parallel = parallel
        self.bbox = bbox
        self.bbox_offset = bbox_offset
        self.grid_size = grid_size
        self.grid_unit = grid_unit
        self.proj_name = proj_name
        self.window_mode = window_mode
        # self.resolution = resolution
        # self.unit = resolution_unit
        self.interp_method = interp_method
        self.approximation = approximation
        self.tc = tc
        
        self.grid_size = to_seconds(grid_size, grid_unit)
        
        # Directory setup
        directory_setup(proj_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._validate_params()
        
        
    def _validate_params(self) -> None:
        '''Validate parameters'''
        if self.ellipsoid not in ['wgs84', 'grs80']:
            raise ValueError('Ellipsoid must be \'wgs84\' or \'grs80\'')
        if any(x is None for x in self.bbox):
                raise ValueError('bbox must contain four numbers [W, E, S, N] when input-file is not provided or --grid is used')
        if len(self.bbox) != 4:
            raise ValueError('bbox must contain exactly four numbers [W, E, S, N]')
        min_lon, max_lon, min_lat, max_lat = self.bbox
        if not all(isinstance(x, (int, float)) for x in self.bbox):
            raise ValueError('bbox values must be numbers')
        if not (min_lon <= max_lon and min_lat <= max_lat):
            raise ValueError('Invalid bbox: west must be <= east, south <= north')
        if self.window_mode not in ['radius', 'fixed']:
            print('Warning: Unidentified window_mode specified. Defaulting to "radius"')
            self.window_mode = 'radius'
        if self.topo not in ['srtm30plus', 'srtm', 'cop', 'nasadem', 'gebco']:
            raise ValueError('topo must be one of: srtm30plus, srtm, cop, nasadem, gebco')
        if self.interp_method not in ['linear', 'slinear', 'cubic', 'quintic']:
            raise ValueError('--interpolation-method must be one of: linear, slinear, cubic, quintic')
        if self.grid_unit not in ['degrees', 'minutes', 'seconds']:
            raise ValueError('--grid-unit must be one of: degrees, minutes, seconds')


    def download(self) -> xr.Dataset:
        '''Download DEM'''
        dem = dem4geoid(
            bbox=self.bbox,
            downloads_dir=self.model_dir,
            resolution=self.grid_size,
            model=self.topo,
            bbox_off=self.bbox_offset,
            interp_method=self.interp_method
        )
        return dem
    
    def _initialize_terrain(self) -> None:
        '''Intialize the TerrainQuantities object with the DEM'''
        self.ori_topo = self.download()
        self.ref_topo = xr.open_dataset(self.ref_topo) if self.ref_topo else None
        self.tq = terrain.TerrainQuantities(
            ori_topo=self.ori_topo,
            ref_topo=self.ref_topo,
            radius=self.radius,
            ellipsoid=self.ellipsoid,
            bbox_off=self.bbox_offset,
            sub_grid=self.bbox,
            proj_dir=self.proj_name,
            window_mode=self.window_mode
        )

    def compute_tc(self) -> dict:
        '''Compute terrain correction'''
        print(f'Computing terrain correction with radius={self.radius} km and ellipsoid={self.ellipsoid}')
        result = self.tq.terrain_correction(
            parallel=self.parallel,
            chunk_size=self.chunk_size,
            progress=True
        )
        self.tc = result
        output_file = self.output_dir / f'{self.TASK_CONFIG['terrain-correction']['output']['file']}.nc'
        return {
            'status': 'success',
            'output_file': str(output_file)
        }
        
    def compute_rtm(self) -> dict:
        '''Compute RTM gravity anomaly'''
        if self.ref_topo is None:
            raise ValueError('Reference topography (--ref-topo) is required for RTM anomaly.')
        print(f'Computing RTM anomaly with radius={self.radius} km and ellipsoid={self.ellipsoid}')
        result = self.tq.rtm_anomaly(
            parallel=self.parallel,
            chunk_size=self.chunk_size,
            approximation=self.approximation,
            progress=True,
            tc=self.tc
        )
        output_file = self.output_dir / f'{self.TASK_CONFIG["rtm-anomaly"]["output"]["file"]}.nc'
        return {
            'status': 'success',
            'output_file': str(output_file)
        }
    
    def compute_ind(self) -> dict:
        '''Compute indirect effect'''
        output_file = self.output_dir / f'{self.TASK_CONFIG["indirect-effect"]["output"]["file"]}.nc'
        if output_file.exists():
            print(f'Indirect effect exists. To recompute, please delete existing NetCDF file and rerun. Skipping computation...')
            return {
                'status': 'skipped',
                'output_file': str(output_file)
            }
        else:
            print(f'Computing indirect effect with radius={self.radius} km and ellipsoid={self.ellipsoid}')
            result = self.tq.indirect_effect(
                parallel=self.parallel,
                chunk_size=self.chunk_size,
                progress=True
            )
            
            return {
                'status': 'success',
                'output_file': str(output_file)
            }
    
    def compute_rtm_height(self) -> dict:
        '''Compute RTM height anomaly'''
        if self.ref_topo is None:
            raise ValueError('Reference topography (--ref-topo) is required for RTM height anomaly.')
        print(f'Computing RTM height anomaly with radius={self.radius} km and ellipsoid={self.ellipsoid}')
        result = self.tq.rtm_height_anomaly(
            parallel=self.parallel,
            chunk_size=self.chunk_size,
            progress=True
        )
        output_file = self.output_dir / f'{self.TASK_CONFIG["height-anomaly"]["output"]["file"]}.nc'
        return {
            'status': 'success',
            'output_file': str(output_file)
        }
        
    def compute_site(self) -> dict:
        '''Compute secondary indirect topographic effect on gravity'''
        # Initialize tq if not hasattr(self, 'tq'):
        if not hasattr(self, 'tq'):
            self._initialize_terrain()
            
        result = self.tq.secondary_indirect_effect()
        output_file = self.output_dir / f'{self.TASK_CONFIG["site"]["output"]["file"]}.nc'
        return {
            'status': 'success',
            'output_file': str(output_file)
        }


    def run(self, tasks: list) -> dict:
        '''Execute the specified tasks.'''
        if not hasattr(self, 'tq'):
            self._initialize_terrain()

        results = {}
        for task in tasks:
            if task not in self.TASK_CONFIG:
                raise ValueError(f'Unknown task: {task}')
            method = getattr(self, self.TASK_CONFIG[task]['method'])
            results[task] = method()
        output_files = [result['output_file'] for result in results.values() if result.get('output_file')]
        return {'status': 'success', 'output_files': output_files}

def add_topo_arguments(parser) -> None:
    parser.add_argument('--topo', type=str, required=True, choices=['srtm30plus', 'srtm', 'cop', 'nasadem', 'gebco'], 
                        help='DEM model (e.g., srtm, srtm30plus, cop, nasadem, gebco)')
    parser.add_argument('-b', '--bbox', type=float, nargs=4, required=True, 
                        help='Bounding box [W, E, S, N] in degrees')
    parser.add_argument('--ref-topo', type=str, 
                        help='Path to reference elevation file (required for residual terrain quantities)')
    parser.add_argument('-md', '--model-dir', type=str, default=None, 
                        help='Directory for DEM files')
    parser.add_argument('--radius', type=float, default=110.0, 
                        help='Search radius in kilometers. Default: 110 km')
    parser.add_argument('-ell', '--ellipsoid', type=str, default='wgs84', choices=['wgs84', 'grs80'], 
                        help='Reference ellipsoid. Default: wgs84')
    parser.add_argument('--do', type=str, default='all', choices=['download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site', 'all'], 
                        help='Computation steps to perform')
    parser.add_argument('-s', '--start', type=str, choices=['download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site'],
                        help='Start processing from this step')
    parser.add_argument('-e', '--end', type=str, choices=['download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site'],
                        help='End processing at this step')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject', 
                        help='Name of the project directory')
    parser.add_argument('-bo', '--bbox-offset', type=float, default=1.0, 
                        help='Offset around bounding box in degrees')
    parser.add_argument('-gs', '--grid-size', type=float, default=30, 
                        help='Grid size (resolution) in degrees, minutes, or seconds. (Default: 30 seconds)')
    parser.add_argument('-gu', '--grid-unit', type=str, default='seconds', choices=['degrees', 'minutes', 'seconds'], 
                        help='Unit of grid size. Dafault: seconds')
    parser.add_argument('--window-mode', type=str, default='radius', choices=['radius', 'fixed'], 
                        help='Method for selecting sub-grid for computation. Default: radius')
    parser.add_argument('-p', '--parallel', action='store_true', default=False, 
                        help='Enable parallel processing')
    parser.add_argument('--chunk-size', type=int, default=500, 
                        help='Chunk size for parallel processing')
    parser.add_argument('--interpolation-method', type=str, default='slinear', choices=['linear', 'nearest', 'slinear', 'cubic', 'quintic'],
                        help='Interpolation method to resample the DEM to --resolution. Default: slinear')

def main(args=None) -> int:
    '''
    Main function to parse arguments and run topographic quantities computation.
    '''
    if args is None:
        parser = argparse.ArgumentParser(
            description=(
            'Calculate topographic quantities from DEM.'
            'Supported tasks: download, terrain-correction, rtm-anomaly, indirect-effect, height-anomaly.'
            )
        )
        add_topo_arguments(parser)
        args = parser.parse_args()

    # Define workflow
    workflow = ['download', 'terrain-correction', 'rtm-anomaly', 'indirect-effect', 'height-anomaly', 'site']
    # Determine tasks to execute
    if args.do != 'all' and (args.start or args.end):
        raise ValueError('Cannot specify both --do and --start or --end.')
    if args.do == 'all':
        tasks = [t for t in workflow if t != 'download'] # Exclude 'download' from all
    elif args.start or args.end:
        start_idx = 0 if args.start is None else workflow.index(args.start)
        end_idx = len(workflow) if args.end is None else workflow.index(args.end) + 1
        tasks = workflow[start_idx:end_idx + 1]
        if 'download' in tasks:
            tasks.remove('download') # Download is handled implicitly
    else:
        tasks = [args.do]
        if args.do == 'download':
            tasks = [] # Download is handled implicitly

    # Initialize and run workflow
    topo_workflow = TopographicQuantities(
        topo=args.topo,
        ref_topo=args.ref_topo,
        model_dir=args.model_dir,
        output_dir=Path(args.proj_name) / 'results',
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        radius=args.radius,
        proj_name=args.proj_name,
        bbox=args.bbox,
        bbox_offset=args.bbox_offset,
        grid_size=args.grid_size,
        grid_unit=args.grid_unit,
        window_mode=args.window_mode,
        parallel=args.parallel,
        interp_method=args.interpolation_method
    )
    
    # Ensure DEM is available before running tasks
    topo_workflow._initialize_terrain()
    
    # Run tasks if any
    if tasks:
        result = topo_workflow.run(tasks)
        print(f'Completed tasks: {", ".join(tasks)}')
        print(f'Output files: {", ".join(result["output_files"])}')
    else:
        print('No computation tasks specified.')
        
    return 0

if __name__ == '__main__':
    sys.exit(main())