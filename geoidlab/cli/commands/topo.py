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
from datetime import datetime
from tzlocal import get_localzone

from geoidlab.cli.commands.utils.common import directory_setup, to_seconds
from geoidlab import terrain
from geoidlab import curtin
from geoidlab.dem import dem4geoid
from geoidlab import constants
from geoidlab.utils.io import apply_ellipsoid_attrs

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
            'output': {'key': 'dg_RTM', 'file': 'RTM'},  # Change from 'dg_RTM' to 'rtm'
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
        },
        'atm-corr': {
            'method': 'compute_atm_corr',
            'terrain_method': 'atm_correction_grid',
            'output': {'key': 'Dg_atm', 'file': 'Dg_atm'}
        },
    }
    
    def __init__(
        self,
        topo: str = None,
        topo_file: str | Path = None,
        topo_url: str = None,
        topo_cog_url: str = None,
        topo_lon_name: str = 'x',
        topo_lat_name: str = 'y',
        topo_height_name: str = 'z',
        ref_topo: str = None,
        dtm_model: str | Path = None,
        earth2014_model: str | None = None,
        earth2014_resolution: str = '5min',
        dtm_nmax: int = 360,
        dtm_chunk_size: int = 200,
        model_dir: str | Path = None,
        output_dir: str | Path = 'results',
        ellipsoid: str = 'wgs84',
        ellipsoid_name: str | None = None,
        chunk_size: int = 500,
        radius: float = 167.,
        proj_name: str = 'GeoidProject',
        bbox: list[float] = None,
        bbox_offset: float = 2.0,
        grid_size: float = None,
        grid_unit: str = 'seconds',
        window_mode: str = 'radius',
        parallel: bool = False,
        # resolution: int = 30,
        # resolution_unit: str = 'seconds',
        interp_method: str = 'slinear',
        approximation: bool = False,
        tc: xr.Dataset = None,
        atm_method: str = 'noaa',
        constant_density: bool = True,
        density_model: str | None = '30s',
        density_resolution: int | None = None,
        density_resolution_unit: str | None = None,
        density_file: str | Path | None = None,
        density_interp_method: str = 'nearest',
        density_unit: str = 'kg/m3',
        density_save: bool = True,
        tasks: list[str] = None,
    ) -> None:
        '''
        
        Parameters
        ----------
        topo           : Path to DEM file
        ref_topo       : Path to reference topo file
        dtm_nmax       : Maximum degree for DTM spherical harmonic synthesis
        dtm_chunk_size : Chunk size for DTM spherical harmonic synthesis
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
        atm_method     : Atmospheric correction method. Options: 'noaa', 'ngi', 'wenzel'
        tasks          : List of tasks to perform. Options: 'download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site', 'atm-corr'
        
        Returns
        -------
        None
        '''
        self.topo = topo
        self.topo_file = Path(topo_file) if topo_file is not None else None
        self.topo_url = topo_url
        self.topo_cog_url = topo_cog_url
        self.topo_lon_name = topo_lon_name
        self.topo_lat_name = topo_lat_name
        self.topo_height_name = topo_height_name
        self.ref_topo = ref_topo
        self.dtm_model = Path(dtm_model) if dtm_model is not None else None
        self.earth2014_model = earth2014_model
        self.earth2014_resolution = earth2014_resolution
        self.dtm_nmax = dtm_nmax
        self.dtm_chunk_size = dtm_chunk_size
        self.model_dir = Path(model_dir) if model_dir else Path(proj_name) / 'downloads'
        self.output_dir = Path(output_dir)
        self.ellipsoid = ellipsoid
        self.ellipsoid_name = ellipsoid_name
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
        self.atm_method = atm_method
        self.constant_density = constant_density
        self.density_model = density_model
        self.density_resolution = density_resolution
        self.density_resolution_unit = density_resolution_unit
        self.density_file = density_file
        self.density_interp_method = density_interp_method
        self.density_unit = density_unit
        self.density_save = density_save
        self.tasks = tasks or []
        
        self.grid_size = to_seconds(grid_size, grid_unit)
        
        # Directory setup
        directory_setup(proj_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._validate_params()
        
        
    def _validate_params(self) -> None:
        '''Validate parameters'''
        constants.resolve_ellipsoid(self.ellipsoid)
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
        topo_sources = [self.topo is not None, self.topo_file is not None, self.topo_url is not None, self.topo_cog_url is not None]
        if sum(topo_sources) != 1:
            raise ValueError(
                'Specify exactly one DEM source: --topo, --topo-file, --topo-url, or --topo-cog-url.'
            )
        if self.topo is not None and self.topo not in ['srtm30plus', 'srtm', 'cop', 'nasadem', 'gebco']:
            raise ValueError('topo must be one of: srtm30plus, srtm, cop, nasadem, gebco')
        if self.dtm_model is not None and self.earth2014_model is not None:
            raise ValueError('Specify at most one reference-topography source: --dtm-model or --earth2014-model.')
        if self.interp_method not in ['linear', 'slinear', 'cubic', 'quintic']:
            raise ValueError('--interpolation-method must be one of: linear, slinear, cubic, quintic')
        if self.grid_unit not in ['degrees', 'minutes', 'seconds']:
            raise ValueError('--grid-unit must be one of: degrees, minutes, seconds')


    def download(self) -> xr.Dataset:
        '''Download DEM'''
        dem = dem4geoid(
            bbox=self.bbox,
            ncfile=self.topo_file,
            url=self.topo_url,
            cog_url=self.topo_cog_url,
            downloads_dir=self.model_dir,
            resolution=self.grid_size,
            model=self.topo,
            bbox_off=self.bbox_offset,
            interp_method=self.interp_method,
            lon_name=self.topo_lon_name,
            lat_name=self.topo_lat_name,
            height_name=self.topo_height_name,
        )
        return dem
    
    def _get_reference_topography(self) -> xr.Dataset:
        '''Generate reference topography from DTM2006.0 if not provided'''
        if self.ref_topo is None:
            from geoidlab.dtm import DigitalTerrainModel
            dtm_model_path = self.dtm_model
            model_desc = 'DTM2006.0'
            if dtm_model_path is None and self.earth2014_model is not None:
                dtm_model_path = curtin.download_shc_model(
                    model=self.earth2014_model,
                    resolution=self.earth2014_resolution,
                    output_dir=self.model_dir,
                )
                model_desc = dtm_model_path.name
            elif dtm_model_path is not None:
                model_desc = dtm_model_path.name

            print(f'Generating reference topography from {model_desc} up to degree {self.dtm_nmax}...')
            
            # Create coordinates matching original topography
            ori_x = self.ori_topo['x'].values
            ori_y = self.ori_topo['y'].values
            lon, lat = np.meshgrid(ori_x, ori_y)
            
            # Initialize DTM object and compute heights
            dtm = DigitalTerrainModel(
                model_name=dtm_model_path,
                nmax=self.dtm_nmax,
                ellipsoid=self.ellipsoid
            )
            H = dtm.dtm2006_height(lon=lon, lat=lat, chunk_size=self.dtm_chunk_size, save=False)
            # Create xarray Dataset
            ref_topo = xr.Dataset(
                {
                    'z': (('y', 'x'), H)
                },
                coords={
                    'x': (('x',), ori_x),
                    'y': (('y',), ori_y)
                }
            )
            ref_topo.attrs['description'] = f'Reference topography from {model_desc} up to degree {self.dtm_nmax}'
            
            # Add standard attributes
            local_tz = get_localzone()
            date_created = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
            ref_topo.attrs.update({
                'date_created': f'{date_created} {local_tz}',
                'created_by'  : 'GeoidLab',
                'website'     : 'https://github.com/cikelly/geoidlab',
                'copyright'   : f'Copyright (c) {datetime.now().year}, Caleb Kelly',
            })
            apply_ellipsoid_attrs(ref_topo, ellipsoid=self.ellipsoid, ellipsoid_name=self.ellipsoid_name)
            
            # Save for future use
            model_stem = Path(model_desc).stem.replace('.', '_')
            out_file = Path(self.proj_name) / 'downloads' / f'{model_stem}_nmax{self.dtm_nmax}.nc'
            ref_topo.to_netcdf(out_file)
            print(f'Saved reference topography to {out_file}')
            
            return ref_topo
        else:
            return xr.open_dataset(self.ref_topo)
                
    
    def _initialize_terrain(self) -> None:
        '''Intialize the TerrainQuantities object with the DEM'''
        self.ori_topo = self.download()
        
        # Check if RTM tasks are requested
        needs_ref_topo = any(task in ['rtm-anomaly', 'height-anomaly'] for task in self.tasks)
        
        # Get reference topogarphy if needed for RTM tasks
        if needs_ref_topo:
            self.ref_topo = self._get_reference_topography()
        # elif self.ref_topo:
        #     self.ref_topo = xr.open_dataset(self.ref_topo)
        # self.ref_topo = xr.open_dataset(self.ref_topo) if self.ref_topo else None
        
        self.tq = terrain.TerrainQuantities(
            ori_topo=self.ori_topo,
            ref_topo=self.ref_topo,
            radius=self.radius,
            ellipsoid=self.ellipsoid,
            bbox_off=self.bbox_offset,
            sub_grid=self.bbox,
            proj_dir=self.output_dir,  # Change from proj_name to output_dir
            window_mode=self.window_mode,
            density_download_dir=self.model_dir,
            constant_density=self.constant_density,
            density_model=self.density_model,
            density_resolution=self.density_resolution,
            density_resolution_unit=self.density_resolution_unit,
            density_file=self.density_file,
            density_interp_method=self.density_interp_method,
            density_unit=self.density_unit,
            density_save=self.density_save,
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
        output_file = self.output_dir / f"{self.TASK_CONFIG['terrain-correction']['output']['file']}.nc"
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
        output_file = self.output_dir / f"{self.TASK_CONFIG['rtm-anomaly']['output']['file']}.nc"
        return {
            'status': 'success',
            'output_file': str(output_file)
        }
    
    def compute_ind(self) -> dict:
        '''Compute indirect effect'''
        output_file = self.output_dir / f"{self.TASK_CONFIG['indirect-effect']['output']['file']}.nc"
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
        output_file = self.output_dir / f"{self.TASK_CONFIG['height-anomaly']['output']['file']}.nc"
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
        output_file = self.output_dir / f"{self.TASK_CONFIG['site']['output']['file']}.nc"
        return {
            'status': 'success',
            'output_file': str(output_file)
        }

    def compute_atm_corr(self) -> dict:
        '''Compute atmospheric correction over DEM grid'''
        # Initialize tq if not hasattr(self, 'tq')
        if not hasattr(self, 'tq'):
            self._initialize_terrain()
            
        atm_corr = self.tq.atm_correction_grid(method=self.atm_method)
        output_file = self.output_dir / f"{self.TASK_CONFIG['atm-corr']['output']['file']}.nc"
        
        return {'status': 'success', 'output_file': str(output_file)}

    def compute_corrections(self) -> dict:
        '''Compute both atmospheric and SITE corrections efficiently'''
        atm_file = self.output_dir / 'Dg_atm.nc'
        site_file = self.output_dir / 'Dg_SITE.nc'
        # Only compute if not already present
        if not atm_file.exists() or not site_file.exists():
            if not hasattr(self, 'tq'):
                self._initialize_terrain()
            if not atm_file.exists():
                atm_corr = self.tq.atm_correction_grid()
                output_file = self.output_dir / f"{self.TASK_CONFIG['atm-corr']['output']['file']}.nc"
            if not site_file.exists():    
                site = self.tq.secondary_indirect_effect()
                output_file = self.output_dir / f"{self.TASK_CONFIG['site']['output']['file']}.nc"
        return {'status': 'success', 'output_files': [str(atm_file), str(site_file)]}

    def run(self, tasks: list) -> dict:
        '''Execute the specified tasks.'''
        if not hasattr(self, 'tq'):
            self._initialize_terrain()

        results = {}
        task_queue = list(tasks)
        if task_queue:
            print(f'Running tasks {task_queue} sequentially.')
        for idx, task in enumerate(task_queue):
            if task not in self.TASK_CONFIG:
                raise ValueError(f'Unknown task: {task}')
            method = getattr(self, self.TASK_CONFIG[task]['method'])
            results[task] = method()
            remaining_tasks = task_queue[idx + 1:]
            print(f'Remaining tasks {remaining_tasks}.')
        output_files = [result['output_file'] for result in results.values() if result.get('output_file')]
        return {'status': 'success', 'output_files': output_files}

def add_topo_arguments(parser) -> None:
    parser.add_argument('--topo', type=str,
                        help='Built-in DEM model. Options: srtm30plus, srtm, cop, nasadem, gebco')
    parser.add_argument('--topo-file', type=str,
                        help='Path to a local DEM file to use instead of a built-in model.')
    parser.add_argument('--topo-url', type=str,
                        help='URL to a custom remote DEM file that should be downloaded before ingestion.')
    parser.add_argument('--topo-cog-url', type=str,
                        help='URL to a cloud-optimized or GDAL-readable remote DEM source.')
    parser.add_argument('--topo-lon-name', type=str, default='x',
                        help='Longitude coordinate name for a user-supplied DEM file. Default: x')
    parser.add_argument('--topo-lat-name', type=str, default='y',
                        help='Latitude coordinate name for a user-supplied DEM file. Default: y')
    parser.add_argument('--topo-height-name', type=str, default='z',
                        help='Height/elevation variable name for a user-supplied DEM file. Default: z')
    parser.add_argument('-b', '--bbox', type=float, nargs=4, required=True, 
                        help='Bounding box [W, E, S, N] in degrees')
    parser.add_argument('--ref-topo', type=str, 
                        help='Path to reference elevation file (required for residual terrain quantities)')
    parser.add_argument('--dtm-nmax', type=int, default=360,
                        help='Maximum degree for DTM2006.0 when computing reference topography. Default: 360')
    parser.add_argument('--dtm-model', type=str, default=None,
                        help='Optional path to custom terrain SHC model file used for reference topography synthesis (.xz text or Earth2014 .bshc).')
    parser.add_argument('--earth2014-model', type=str, default=None, choices=['sur', 'bed', 'tbi', 'ret', 'ice'],
                        help='Optional Earth2014 relief model to download and use for reference topography synthesis. Choices: sur, bed, tbi, ret, ice.')
    parser.add_argument('--earth2014-resolution', type=str, default='5min', choices=['5min', '1min'],
                        help='Earth2014 SHC family used with --earth2014-model. 5min=degree2160, 1min=degree10800. Default: 5min')
    parser.add_argument('--dtm-chunk-size', type=int, default=200,
                        help='Chunk size for DTM2006.0 spherical harmonic synthesis. Use smaller chunks for larger grids or large dtm-nmax. Default: 200')
    parser.add_argument('-md', '--model-dir', type=str, default=None, 
                        help='Directory for DEM files')
    parser.add_argument('--radius', type=float, default=167.0, 
                        help='Search radius in kilometers. Default: 167 km')
    parser.add_argument('-ell', '--ellipsoid', type=str, default='wgs84', 
                        help='Reference ellipsoid: wgs84, grs80, or JSON object string')
    parser.add_argument('--ellipsoid-name', type=str, default=None,
                        help='Optional ellipsoid name to store in output metadata')
    parser.add_argument('--do', type=str, default=None, choices=['download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site', 'atm-corr', 'all'], 
                        help='Computation steps to perform.')
    parser.add_argument('-s', '--start', type=str, choices=['download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site', 'atm-corr'],
                        help='Start processing from this step')
    parser.add_argument('-e', '--end', type=str, choices=['download', 'terrain-correction', 'indirect-effect', 'rtm-anomaly', 'height-anomaly', 'site', 'atm-corr'],
                        help='End processing at this step')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject', 
                        help='Name of the project directory')
    parser.add_argument('-bo', '--bbox-offset', type=float, default=2.0, 
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
    parser.add_argument('--atm-method', type=str, default='noaa', choices=['noaa', 'ngi', 'wenzel'],
                        help='Atmospheric correction method. Default: noaa')
    parser.add_argument('--variable-density', action='store_true', default=False,
                        help='Enable variable topographic density (UNB model ingestion) instead of constant crustal density.')
    parser.add_argument('--density-model', type=str, default='30s',
                        help='UNB density model key/file to ingest. Default: 30s')
    parser.add_argument('--density-resolution', type=int, default=None,
                        help='Optional density model resolution value used with --density-resolution-unit.')
    parser.add_argument('--density-resolution-unit', type=str, default=None, choices=['d', 'm', 's'],
                        help='Density model resolution unit: d, m, or s.')
    parser.add_argument('--density-file', type=str, default=None,
                        help='Path to preprocessed density NetCDF file. If provided, download/ingest is skipped.')
    parser.add_argument('--density-interp-method', type=str, default='nearest',
                        choices=['linear', 'nearest', 'slinear', 'cubic', 'quintic'],
                        help='Interpolation method for aligning density to DEM grid. Default: nearest')
    parser.add_argument('--density-unit', type=str, default='kg/m3', choices=['kg/m3', 'g/cm3'],
                        help='Density unit used internally after ingestion. Default: kg/m3')
    parser.add_argument('--no-density-save', dest='density_save', action='store_false', default=True,
                        help='Disable saving processed density NetCDF cache file in the download/model directory.')

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
    workflow = ['download', 'terrain-correction', 'rtm-anomaly', 'indirect-effect', 'height-anomaly', 'site', 'atm-corr']
    # Determine tasks to execute
    if args.do is not None and (args.start or args.end):
        raise ValueError('Cannot specify both --do and --start or --end.')
    if args.start or args.end:
        start_idx = 0 if args.start is None else workflow.index(args.start)
        end_idx = len(workflow) if args.end is None else workflow.index(args.end) + 1
        tasks = workflow[start_idx:end_idx + 1]
        if 'download' in tasks:
            tasks.remove('download') # Download is handled implicitly
    elif args.do == 'all':
        tasks = [t for t in workflow if t != 'download'] # Exclude 'download' from all
    elif args.do is not None:
        tasks = [args.do]
        if args.do == 'download':
            tasks = [] # Download is handled implicitly
    else:
        tasks = [t for t in workflow if t != 'download']

    # Initialize and run workflow
    topo_workflow = TopographicQuantities(
        topo=args.topo,
        topo_file=args.topo_file,
        topo_url=args.topo_url,
        topo_cog_url=args.topo_cog_url,
        topo_lon_name=args.topo_lon_name,
        topo_lat_name=args.topo_lat_name,
        topo_height_name=args.topo_height_name,
        ref_topo=args.ref_topo,
        dtm_model=args.dtm_model,
        earth2014_model=args.earth2014_model,
        earth2014_resolution=args.earth2014_resolution,
        dtm_nmax=args.dtm_nmax,
        dtm_chunk_size=args.dtm_chunk_size,
        model_dir=args.model_dir,
        output_dir=Path(args.proj_name) / 'results',
        ellipsoid=args.ellipsoid,
        ellipsoid_name=args.ellipsoid_name,
        chunk_size=args.chunk_size,
        radius=args.radius,
        proj_name=args.proj_name,
        bbox=args.bbox,
        bbox_offset=args.bbox_offset,
        grid_size=args.grid_size,
        grid_unit=args.grid_unit,
        window_mode=args.window_mode,
        parallel=args.parallel,
        interp_method=args.interpolation_method,
        constant_density=not args.variable_density,
        density_model=args.density_model,
        density_resolution=args.density_resolution,
        density_resolution_unit=args.density_resolution_unit,
        density_file=args.density_file,
        density_interp_method=args.density_interp_method,
        density_unit=args.density_unit,
        density_save=args.density_save,
        tasks=tasks
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
