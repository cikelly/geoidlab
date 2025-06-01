############################################################
# Gravity Reduction CLI interface                          #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import pandas as pd
import xarray as xr
import numpy as np

from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from geoidlab.cli.commands.utils.common import directory_setup, to_seconds
from geoidlab.utils.interpolators import Interpolators
from geoidlab import gravity
from geoidlab.cli.commands.topo import TopographicQuantities
from geoidlab.tide import GravityTideSystemConverter
from geoidlab.icgem import get_ggm_tide_system
from geoidlab.utils.io import save_to_netcdf


def decimate_data(df: pd.DataFrame, n_points: int, verbose: bool = False) -> pd.DataFrame:
    '''
    Decimate a DataFrame to a specified number of points using KMeans clustering.
    
    Parameters
    ----------
    df        : Marine data with 'lon', 'lat', and 'Dg'.
    n_points  : Number of points to retain after decimation.
    verbose   : If True, print information about the decimation process.
    
    Returns
    -------
    Decimated DataFrame with n_points rows.
    '''
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    import numpy as np
    
    if n_points >= len(df):
        if verbose:
            print(f'Requested {n_points} points, but DataFrame has {len(df)} points. Skipping decimation...')
        return df
    
    if n_points < 10:
        raise ValueError(f'Requested number of points ({n_points}) is too low. Must be >= 10.')
    
    coords = np.column_stack((df['lon'], df['lat']))
    if verbose:
        print(f'Decimating marine gravity data from {len(df)} to {n_points} points using KMeans clustering...')
    
    kmeans = KMeans(n_clusters=n_points, random_state=42, n_init=10)
    kmeans.fit(coords)
    centers = kmeans.cluster_centers_
    distances = cdist(coords, centers)
    indices = np.argmin(distances, axis=0)
    
    decimated_df = df.iloc[indices].copy()
    if verbose:
        print(f'Decimation complete. Retained {len(decimated_df)} points.')
    
    return decimated_df

class GravityReduction:
    '''Class to perform gravity reductions (Free-air, Bouguer, helmert/Helmert anomalies)'''
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
        'helmert': {
            'method': 'compute_helmert',
            'output': {'key': 'helmert', 'file': 'helmert'}
        }
    }
    
    def __init__(
        self,
        input_file: str,
        model: str = None,
        model_dir: str | Path = None,
        marine_data: str = None,
        gravity_tide: str = None,
        ellipsoid: str = 'wgs84',
        converted: bool = False,
        grid: bool = False,
        grid_size: float = None,
        grid_unit: str = 'seconds',
        grid_method: str = 'linear',
        bbox: list = [None, None, None, None],
        bbox_offset: float = 1.0,
        proj_name: str = 'GeoidProject',
        topo: str = None,
        tc_file: str = None,
        radius: float = 110.0,
        interp_method: str = 'slinear',
        parallel: bool = False,
        chunk_size: int = 500,
        atm: bool = False,
        atm_method: str = 'noaa',
        ellipsoidal_correction: bool = False,
        window_mode: str = 'radius',
        tc_grid_size: float = 30.0,
        decimate: bool = False,
        decimate_threshold: int = 600,
        site: bool = False,
        max_deg: int = 90
    ) -> None:
        self.input_file = input_file
        self.model = model
        self.model_dir = Path(model_dir) if model_dir else Path(proj_name) / 'downloads'
        self.marine_data = marine_data
        self.gravity_tide = gravity_tide
        self.ellipsoid = ellipsoid
        self.converted = converted
        self.grid = grid
        self.grid_size = grid_size
        self.grid_unit = grid_unit
        self.grid_method = grid_method
        self.bbox = bbox
        self.bbox_offset = bbox_offset
        self.proj_name = proj_name
        self.topo = topo
        self.tc_file = tc_file
        self.radius = radius
        self.interp_method = interp_method
        self.parallel = parallel
        self.chunk_size = chunk_size
        self.output_dir = Path(proj_name) / 'results'
        self.lonlatheight = None
        self.free_air = None
        self.bouguer = None
        self.tc = None
        self.atm = atm
        self.atm_method = atm_method
        self.ellipsoidal_correction = ellipsoidal_correction
        self.window_mode = window_mode
        self.tc_grid_size = tc_grid_size
        self.decimate = decimate
        self.decimate_threshold = decimate_threshold
        self.ggm_tide = None
        self.site = site
        self.max_deg = max_deg

        directory_setup(proj_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
                
        self._validate_params()

    def _validate_params(self) -> None:
        if not Path(self.input_file).is_file():
            raise ValueError(f'Input file {self.input_file} does not exist')
        if self.marine_data and not Path(self.marine_data).is_file():
            raise ValueError(f'Marine data file {self.marine_data} does not exist')
        if self.gravity_tide and self.gravity_tide not in ['mean_tide', 'zero_tide', 'tide_free']:
            raise ValueError('Gravity tide must be mean_tide, zero_tide, or tide_free')
        if self.ellipsoid not in ['wgs84', 'grs80']:
            raise ValueError('Ellipsoid must be wgs84 or grs80')
        if self.gravity_tide and self.gravity_tide not in ['mean_tide', 'zero_tide', 'tide_free']:
            raise ValueError('Gravity tide must be mean_tide, zero_tide, or tide_free')
        if self.grid:
            if not (self.grid_size and self.bbox):
                raise ValueError('grid-size and bbox are required when --grid is used')
            min_lon, max_lon, min_lat, max_lat = self.bbox
            if len(self.bbox) != 4 or not all(isinstance(x, (int, float)) for x in self.bbox):
                raise ValueError('bbox must contain four numbers [W, E, S, N]')
            if not (min_lon <= max_lon and min_lat <= max_lat):
                raise ValueError('Invalid bbox: west must be <= east, south <= north')
        if self.topo and self.topo not in ['srtm30plus', 'srtm', 'cop', 'nasadem', 'gebco']:
            raise ValueError('topo must be one of: srtm30plus, srtm, cop, nasadem, gebco')
        if self.tc_file and not Path(self.tc_file).is_file():
            raise ValueError(f'Terrain correction file {self.tc_file} does not exist')
        if self.grid_unit not in ['degrees', 'minutes', 'seconds']:
            raise ValueError('--grid-unit must be one of: degrees, minutes, seconds')
        if self.interp_method not in ['linear', 'slinear', 'cubic', 'quintic']:
            raise ValueError('--interpolation-method must be one of: linear, slinear, cubic, quintic')
        if self.grid_method not in ['linear', 'spline', 'kriging', 'rbf', 'idw', 'biharmonic', 'gpr', 'lsc']:
            raise ValueError("--grid-method must be one of: ['linear', 'spline', 'kriging', 'rbf', 'idw', 'biharmonic', 'gpr', 'lsc']")
        if self.gravity_tide and not self.model:
            raise ValueError('A GGM model must be specified with --model when --gravity-tide is used')
        if self.decimate and self.decimate_threshold < 10:
            raise ValueError('decimate_threshold must be >= 10')
        
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
                self.marine_data = pd.read_csv(marine_path)
            elif marine_path.suffix in ['.xlsx', '.xls']:
                self.marine_data = pd.read_excel(marine_path)
            elif marine_path.suffix == '.txt':
                self.marine_data = pd.read_csv(marine_path, delimiter='\t')
            else:
                raise ValueError(f'Unsupported file format: {marine_path.suffix}')
        if self.marine_data is not None and not all(col in self.marine_data.columns for col in ['lon', 'lat', 'height', 'Dg']):
            raise ValueError('Marine data must contain columns: lon, lat, height, and Dg')
        
    def _convert_tide_system(self) -> Path | None:
        '''Convert input data to GGM tide system.'''
        if not self.converted and self.gravity_tide and self.model:
            model_path = (self.model_dir / self.model).with_suffix('.gfc')
            ggm_tide = get_ggm_tide_system(icgem_file=model_path, model_dir=self.model_dir)
            if ggm_tide != self.gravity_tide:
                print(f'Converting input data from {self.gravity_tide} to {ggm_tide} system...')
                converter = GravityTideSystemConverter(data=self.lonlatheight)
                # Map tide system pairs to conversion methods
                conversion_map = {
                    ('mean_tide', 'tide_free'): 'mean2free',
                    ('tide_free', 'mean_tide'): 'free2mean',
                    ('mean_tide', 'zero_tide'): 'mean2zero',
                    ('zero_tide', 'mean_tide'): 'zero2mean',
                    ('zero_tide', 'tide_free'): 'zero2free',
                    ('tide_free', 'zero_tide'): 'free2zero'
                }
                conversion_key = (self.gravity_tide, ggm_tide)
                if conversion_key not in conversion_map:
                    raise ValueError(f'No conversion defined from {self.gravity_tide} to {ggm_tide}')
                
                # Perform conversion
                conversion_method = getattr(converter, conversion_map[conversion_key])
                converted_data = conversion_method()

                # Map GGM tide system to column suffix
                tide_suffix_map = {
                    'tide_free': 'free',
                    'mean_tide': 'mean',
                    'zero_tide': 'zero'
                }
                
                target_suffix = tide_suffix_map[ggm_tide]
                self.lonlatheight['height'] = converted_data[f'height_{target_suffix}']
                if 'gravity' in converted_data.columns:
                    self.lonlatheight['gravity'] = converted_data[f'g_{target_suffix}']
                
                input_filename = Path(self.input_file).stem if self.input_file else 'lonlatheight'
                converted_data_path = self.output_dir / f'{input_filename}_{ggm_tide}.csv'
                self.lonlatheight.to_csv(converted_data_path, index=False)
                print(f'Converted data saved to {converted_data_path}')
                return converted_data_path
            else:
                print('Surface gravity and GGM have the same tide system. Skipping conversion.')
        return None
    
    def _compute_terrain_correction(self) -> xr.Dataset:
        '''Compute or load terrain correction grid.'''
        if self.tc_file:
            print(f'Loading terrain correction from {self.tc_file}')
            tc_grid = xr.open_dataset(self.tc_file)
        else:
            tc_file = self.output_dir / 'TC.nc'
            if tc_file.exists():
                print(f'Loading terrain correction from {tc_file}')
                tc_grid = xr.open_dataset(tc_file)
                return tc_grid
            
            print(f'Computing terrain correction using {self.topo} DEM...')
            topo_workflow = TopographicQuantities(
                topo=self.topo,
                model_dir=self.model_dir,
                output_dir=self.output_dir,
                ellipsoid=self.ellipsoid,
                chunk_size=self.chunk_size,
                radius=self.radius,
                proj_name=self.proj_name,
                bbox=self.bbox,
                bbox_offset=self.bbox_offset,
                grid_size=self.tc_grid_size,
                window_mode=self.window_mode,
                parallel=self.parallel,
                interp_method=self.interp_method
            )
            topo_workflow._initialize_terrain()
            result = topo_workflow.run(['terrain-correction'])
            tc_file = result['output_files'][0]
            tc_grid = xr.open_dataset(tc_file)
            self.topo_workflow = topo_workflow
        return tc_grid
    
    def _interpolate_tc(self, tc_grid: xr.Dataset) -> np.ndarray:
        '''Interpolate terrain correction at lonlatheight locations'''
        tc_values = tc_grid['tc'].values
        interpolator = RegularGridInterpolator(
            (tc_grid['lat'].values, tc_grid['lon'].values), 
            tc_values, 
            method=self.interp_method, 
            bounds_error=False, 
            fill_value=0
        )
        points = np.vstack((self.lonlatheight['lat'], self.lonlatheight['lon'])).T
        tc_interpolated = interpolator(points)
        return tc_interpolated
    
    def _compute_secondary_indirect_effect(self) -> np.ndarray:
        '''Compute Secondary Indirect Topographic Effect (SITE) on gravity.'''
        site_file = self.output_dir / 'Dg_SITE.nc'
        if site_file.exists():
            print(f'Loading SITE from {site_file}')
            site_grid = xr.open_dataset(site_file)
            return site_grid
        
        if not hasattr(self, 'topo_workflow'):
            self.topo_workflow = TopographicQuantities(
                topo=self.topo,
                model_dir=self.model_dir,
                output_dir=self.output_dir,
                ellipsoid=self.ellipsoid,
                chunk_size=self.chunk_size,
                radius=self.radius,
                proj_name=self.proj_name,
                bbox=self.bbox,
                bbox_offset=self.bbox_offset,
                grid_size=self.tc_grid_size,
                window_mode=self.window_mode,
                parallel=self.parallel,
                interp_method=self.interp_method
            )
        self.topo_workflow._initialize_terrain()
        site = self.topo_workflow.run(['site'])
        site_file = site['output_files'][0]
        site_grid = xr.open_dataset(site_file)
        return site_grid
        
    def _interpolate_site(self, site_grid: xr.Dataset) -> np.ndarray:
        '''Interpolate SITE at lonlatheight locations'''
        site_values = site_grid['Dg_SITE'].values
        interpolator = RegularGridInterpolator(
            (site_grid['lat'].values, site_grid['lon'].values), 
            site_values, 
            method=self.interp_method, 
            bounds_error=False, 
            fill_value=0
        )
        points = np.vstack((self.lonlatheight['lat'], self.lonlatheight['lon'])).T
        site_interpolated = interpolator(points)
        return site_interpolated
    
    def _compute_ellipsoidal_correction(self) -> xr.Dataset:
        '''Compute or load ellipsoidal correction grid.'''
        from geoidlab.ggm import GlobalGeopotentialModel
        
        if not self.ggm_tide:
            model_path = (self.model_dir / self.model).with_suffix('.gfc')
            ggm_tide = get_ggm_tide_system(icgem_file=model_path, model_dir=self.model_dir)
        
        if not self.model:
            raise ValueError('Please specify a GGM model for ellipsoidal correction.')
        
        ec_file = self.output_dir / 'Dg_ELL.nc'
        if ec_file.exists():
            print(f'Loading ellipsoidal correction from {ec_file}')
            return xr.open_dataset(ec_file)
        
        print(f'Computing ellipsoidal correction using {self.model} model and max-deg={self.max_deg}...')
        # Create grid for computation
        
        grid_size_deg = to_seconds(resolution=self.grid_size, unit=self.grid_unit) / 3600.0
        min_lon, max_lon, min_lat, max_lat = self.bbox
        lon_grid = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / grid_size_deg) + 1)
        lat_grid = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / grid_size_deg) + 1)
        lon, lat = np.meshgrid(lon_grid, lat_grid)
        
        grav_data = pd.DataFrame({
            'lon': lon.flatten(),
            'lat': lat.flatten(),
            'elevation': np.zeros_like(lon.flatten())
        })
        
        ggm = GlobalGeopotentialModel(
            model_name=self.model,
            model_dir=self.model_dir,
            ellipsoid=self.ellipsoid,
            grav_data=grav_data,
            nmax=self.max_deg,
            chunk_size=self.chunk_size
        )
        
        ec = ggm.ellipsoidal_correction(parallel=self.parallel)
        ec_grid = xr.Dataset(
            {'Dg_ELL': (['lat', 'lon'], ec.reshape(lat.shape))},
            coords={'lon': lon_grid, 'lat': lat_grid}
        )
        
        save_to_netcdf(
            data=ec_grid['Dg_ELL'].values,
            lon=lon_grid,
            lat=lat_grid,
            dataset_key='Dg_ELL',
            filepath=ec_file,
            tide_system=self.ggm_tide if self.ggm_tide is None else ggm_tide
        )
        
        print(f'Ellipsoidal correction saved to {ec_file}')
        return ec_grid
    
    def _interpolate_ellipsoidal_correction(self, ec_grid: xr.Dataset) -> np.ndarray:
        '''Interpolate ellipsoidal correction at lonlatheight locations.'''
        from scipy.interpolate import RegularGridInterpolator
        ec_values = ec_grid['Dg_ELL'].values
        interpolator = RegularGridInterpolator(
            (ec_grid['lat'].values, ec_grid['lon'].values),
            ec_values,
            method=self.interp_method,
            bounds_error=False,
            fill_value=0
        )
        points = np.vstack((self.lonlatheight['lat'], self.lonlatheight['lon'])).T
        ec_interpolated = interpolator(points)
        return ec_interpolated
    
    def _grid_anomalies(self, anomalies_dataframes: dict) -> xr.Dataset:
        '''Grid anomalies over bbox with bbox_offset using Interpolators.'''
        grid_size = to_seconds(self.grid_size, self.grid_unit) / 3600.0
        grid_extent = (
            self.bbox[0] - self.bbox_offset,
            self.bbox[1] + self.bbox_offset,
            self.bbox[2] - self.bbox_offset,
            self.bbox[3] + self.bbox_offset
        )
        
        print(f'Gridding anomalies over {grid_extent} with grid size {grid_size} degrees. Interpolation method: {self.grid_method}...')
        
        gridded_data = {}
        for var_name, df in anomalies_dataframes.items():
            interpolator = Interpolators(
                dataframe=df,
                grid_extent=grid_extent,
                resolution=grid_size,
                resolution_unit='degrees',
                data_key='Dg',
                verbose=False
            )
            if self.grid_method == 'kriging': 
                lon, lat, gridded_values, *_ = interpolator.run(method=self.grid_method, merge=True)
            elif self.grid_method == 'lsc':
                lon, lat, gridded_values, *_ = interpolator.run(method=self.grid_method, merge=True, robust_covariance=True, covariance_model='exp')
            elif self.grid_method == 'rbf':
                lon, lat, gridded_values = interpolator.run(method='rbf', function='linear', merge=False) # quintic
            else:
                lon, lat, gridded_values = interpolator.run(method=self.grid_method, merge=True)
                
            gridded_data[var_name] = (['lat', 'lon'], gridded_values)
        
        ds = xr.Dataset(
            gridded_data,
            coords={'lon': lon[0, :], 'lat': lat[:, 0]}
        )
        return ds
    
    def compute_anomalies(self, anomaly_type: str) -> dict:
        '''Compute Free-air and/or Bouguer anomalies'''
        if self.lonlatheight is None:
            self._process_input()
        
        self._convert_tide_system()
        
        self.free_air, self.bouguer = gravity.gravity_anomalies(
            lat=self.lonlatheight['lat'],
            gravity=self.lonlatheight['gravity'],
            elevation=self.lonlatheight['height'],
            ellipsoid=self.ellipsoid,
            atm=self.atm,
            atm_method=self.atm_method,
        )
        
        anomaly_values = self.free_air if anomaly_type == 'free_air' else self.bouguer
        
        output_file_csv = self.output_dir / f'{anomaly_type}.csv'
        df_csv = pd.DataFrame({
            'lon': self.lonlatheight['lon'],
            'lat': self.lonlatheight['lat'],
            anomaly_type: anomaly_values
        })
        df_csv.to_csv(output_file_csv, index=False)
        print(f'{anomaly_type.capitalize()} anomalies written to {output_file_csv}')
        
        output_files = [str(output_file_csv)]
        if self.grid:
            df_grid = pd.DataFrame({
                'lon': self.lonlatheight['lon'],
                'lat': self.lonlatheight['lat'],
                'Dg': anomaly_values
            })
            anomalies_dataframes = {anomaly_type: df_grid}
            gridded_ds = self._grid_anomalies(anomalies_dataframes)
            output_file_nc = self.output_dir / f'Dg_{anomaly_type}.nc'
            save_to_netcdf(
                data=gridded_ds[anomaly_type].values,
                lon=gridded_ds['lon'].values,
                lat=gridded_ds['lat'].values,
                dataset_key=anomaly_type,
                filepath=output_file_nc,
                tide_system=self.ggm_tide
            )
            print(f'Gridded {anomaly_type} anomalies written to {output_file_nc}')
            output_files.append(str(output_file_nc))
        
        return {
            'status': 'success',
            'output_file': output_files
        }
        
    def compute_helmert(self) -> dict:
        '''Compute helmert/Helmert anomalies using Free-air and terrain corrections.'''
        if not (self.topo or self.tc_file):
            raise ValueError('Either --topo or --tc-file must be provided for helmert anomalies')
        
        output_file_csv = self.output_dir / 'helmert.csv'
        output_file_nc = self.output_dir / 'helmert.nc'
        if output_file_csv.exists() and output_file_nc.exists():
            print(f'helmert anomalies already computed at {output_file_csv} and {output_file_nc}. Skipping computation.')
            return {
                'status': 'success',
                'output_file': [str(output_file_csv), str(output_file_nc)]
            }
        
        if self.lonlatheight is None:
            self._process_input()
        
        if self.free_air is None:
            self.compute_anomalies('free_air')
        
        tc_grid = self._compute_terrain_correction()
        self.tc = self._interpolate_tc(tc_grid)
        del tc_grid
        
        helmert = self.free_air + self.tc
        
        # Initialize SITE values
        self.site_values = np.zeros_like(self.free_air)
        self.ec_values   = np.zeros_like(self.free_air)
        
        if self.site:
            print('Secondary Indirect Topographic Effect (SITE) requested and will be computed')
            site_grid = self._compute_secondary_indirect_effect()
            self.site_values = self._interpolate_site(site_grid)
            print('Applying secondary indirect effect to Helmert anomalies...')
        
        if self.ellipsoidal_correction:
            if not self.model:
                raise ValueError('A GGM model must be specified with --model for ellipsoidal correction')
            print('Ellipsoidal correction requested and will be computed')
            ec_grid = self._compute_ellipsoidal_correction()
            self.ec_values = self._interpolate_ellipsoidal_correction(ec_grid)
            print('Applying ellipsoidal correction to Helmert anomalies...')
        
        helmert += self.site_values + self.ec_values
        
        # Save point-wise land anomalies
        output_file_csv = self.output_dir / 'helmert.csv'
        df_land = pd.DataFrame({
            'lon': self.lonlatheight['lon'],
            'lat': self.lonlatheight['lat'],
            'helmert': helmert,
            'free_air': self.free_air,
            'terrain_correction': self.tc,
            'site': self.site_values,
            'ellipsoidal_correction': self.ec_values
        })
        if self.bouguer is not None:
            df_land['bouguer'] = self.bouguer
        df_land.to_csv(output_file_csv, index=False)
        print(f'Helmert anomalies (land) written to {output_file_csv}')
        
        del df_land
        
        # Prepare data for gridding with marine data merge
        land_df = pd.DataFrame({
            'lon': self.lonlatheight['lon'],
            'lat': self.lonlatheight['lat'],
            'Dg': helmert
        })
        if self.marine_data is not None:  # Ensure marine_data is loaded in _process_input
            print('Marine gravity anomalies provided. Combining with Helmert anomalies...')
            marine_df = self.marine_data[['lon', 'lat', 'Dg']]
            
            # Apply decimation if marine data exceeds threshold or --decimate is specified
            if self.decimate:
                if len(marine_df) > self.decimate_threshold:
                    n_points = self.decimate_threshold
                    try:
                        marine_df = decimate_data(marine_df, n_points=n_points, verbose=True)
                    except MemoryError:
                        print(f'Warning: Processing failed due to memory constraints. Try using --decimate and specify number of points to retain using --decimate_threshold.')
                        raise
                    combined_df = pd.concat([land_df, marine_df], ignore_index=True)
            else:
                combined_df = pd.concat([land_df, marine_df], ignore_index=True)
        else:
            combined_df = land_df

        # Handle duplicates
        combined_df = combined_df.drop_duplicates(subset=['lon', 'lat'])
        # Always grid helmert anomalies under 'Dg'
        anomalies_dataframes = {'Dg': combined_df}
        print(f'Shape of anomalies dataframes: {anomalies_dataframes["Dg"].shape}')
        
        import time
        start_time = time.time()
        gridded_ds = self._grid_anomalies(anomalies_dataframes)
        end_time = time.time()
        print(f'Gridding completed in {end_time - start_time} seconds.')
        output_file_nc = self.output_dir / 'Dg.nc'
        save_to_netcdf(
            data=gridded_ds['Dg'].values,
            lon=gridded_ds['lon'].values,
            lat=gridded_ds['lat'].values,
            dataset_key='Dg',
            filepath=output_file_nc,
            tide_system=self.ggm_tide
        )
        print(f'Gridded anomalies written to {output_file_nc}')
        
        output_files = [str(output_file_csv), str(output_file_nc)]
        return {
            'status': 'success',
            'output_file': output_files
        }
        
    def run(self, tasks: list) -> dict:
        '''Execute specified tasks in order.'''
        self.tasks = tasks
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
        output_files = []
        for result in results.values():
            if isinstance(result['output_file'], list):
                output_files.extend(result['output_file'])
            else:
                output_files.append(result['output_file'])
        return {'status': 'success', 'output_files': output_files}


def add_helmert_arguments(parser) -> None:
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Input file with lon, lat, gravity, and height data (required)')
    parser.add_argument('-m', '--model', type=str,
                        help='GGM name (e.g., EGM2008) for tide system alignment')
    parser.add_argument('-md', '--model-dir', type=str, default=None,
                        help='Directory for GGM files')
    parser.add_argument('-n', '--max-deg', type=int, default=90,
                        help='Maximum degree of truncation for ellipsoidal correction')
    parser.add_argument('--marine-data', type=str,
                        help='Input file with lon, lat, height, and Dg.')
    parser.add_argument('--do', type=str, default='helmert', choices=['free-air', 'bouguer', 'helmert', 'all'],
                        help='Computation steps to perform: [free-air, bouguer, helmert, or all (default: helmert)]')
    parser.add_argument('-s', '--start', type=str, choices=['free-air', 'bouguer', 'helmert'],
                        help='Start processing from this step')
    parser.add_argument('-e', '--end', type=str, choices=['free-air', 'bouguer', 'helmert'],
                        help='End processing at this step')
    parser.add_argument('-gt', '--gravity-tide', type=str,
                        help='Tide system of the surface gravity data: [mean_tide, zero_tide, tide_free]')
    parser.add_argument('-g', '--grid', action='store_true',
                        help='Grid the gravity anomalies over a bounding box')
    parser.add_argument('-gs', '--grid-size', type=float,
                        help='Grid size (e.g., 30 for 30 seconds). Required if --grid')
    parser.add_argument('-gu', '--grid-unit', type=str, default='seconds',
                        choices=['degrees', 'minutes', 'seconds'],
                        help='Unit of grid size')
    parser.add_argument('--grid-method', type=str, default='kriging',
                        choices=['linear', 'spline', 'kriging', 'rbf', 'idw', 'biharmonic', 'gpr', 'lsc'],
                        help='Interpolation method for gridding anomalies (default: kriging)')
    parser.add_argument('-b', '--bbox', type=float, nargs=4,
                        help='Bounding box [W, E, S, N] in degrees. Required if --grid')
    parser.add_argument('-bo', '--bbox-offset', type=float, default=1.0,
                        help='Offset around the bounding box in degrees')
    parser.add_argument('-ell', '--ellipsoid', type=str, default='wgs84', choices=['wgs84', 'grs80'],
                        help='Reference ellipsoid')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject',
                        help='Project directory for downloads and results')
    parser.add_argument('-c', '--converted', action='store_true',
                        help='Indicate that input data is already in the target tide system')
    parser.add_argument('--topo', type=str, choices=['srtm30plus', 'srtm', 'cop', 'nasadem', 'gebco'],
                        help='DEM model for terrain correction (required for helmert unless --tc-file is provided)')
    parser.add_argument('--tc-file', type=str,
                        help='Path to precomputed terrain correction NetCDF file')
    parser.add_argument('--radius', type=float, default=110.0,
                        help='Search radius in kilometers for terrain correction')
    parser.add_argument('--interp-method', type=str, default='slinear',
                        choices=['linear', 'slinear', 'cubic', 'quintic'],
                        help='Interpolation method for terrain correction')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing for terrain correction')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Chunk size for parallel processing')
    parser.add_argument('--atm', action='store_true',
                        help='Request atmospheric correction. Default: False')
    parser.add_argument('--atm-method', type=str, default='noaa', choices=['noaa', 'ngi', 'wenzel'],
                        help='Atmospheric correction method. Default: noaa')
    parser.add_argument('--ell-cor', '--ellipsoidal-correction', action='store_true',
                        help='Request ellipsoidal correction. Default: False')
    parser.add_argument('--window-mode', type=str, default='radius', choices=['radius', 'fixed'],
                        help='Method for selecting sub-grid for computation.')
    parser.add_argument('--tc-grid-size', type=float, default=30,
                        help='Grid resolution for computing terrain correction. Keep this in seconds. Default: 30 seconds')
    parser.add_argument('--decimate', action='store_true',
                        help='Decimate marine data. Default observations to retain is 600. Use --decimate-threshold to change this.')
    parser.add_argument('--decimate-threshold', type=int, default=600,
                        help='Threshold for automatic decimation of marine data (default: 600 points).')
    parser.add_argument('--site', action='store_true',
                        help='Apply secondary indirect topographic effect (SITE) on gravity')

def main(args=None) -> None:
    '''
    Main function for gravity reductions. The supported methods are Free-air and Bouguer reductions,
    with outputs including Free-air and Bouguer anomalies, and Helmert/helmert anomalies
    '''
    if args is None:
        parser = argparse.ArgumentParser(
            description=(
                'Perform gravity reduction to compute Free-air, Bouguer, and helmert/Helmert anomalies.'
                'helmert anomalies are the sum of Free-air anomalies and terrain corrections.'
            )
        )
        add_helmert_arguments(parser)
        args = parser.parse_args()
    
    workflow = ['free-air', 'bouguer', 'helmert']
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
        input_file=args.input_file,
        model=args.model,
        model_dir=args.model_dir,
        marine_data=args.marine_data,
        gravity_tide=args.gravity_tide,
        ellipsoid=args.ellipsoid,
        converted=args.converted,
        grid=args.grid,
        grid_size=args.grid_size,
        grid_unit=args.grid_unit,
        grid_method=args.grid_method,
        bbox=args.bbox,
        bbox_offset=args.bbox_offset,
        proj_name=args.proj_name,
        topo=args.topo,
        tc_file=args.tc_file,
        radius=args.radius,
        interp_method=args.interp_method,
        parallel=args.parallel,
        chunk_size=args.chunk_size,
        atm=args.atm,
        atm_method=args.atm_method,
        ellpsoidal_correction=args.ellipsoidal_correction,
        window_mode=args.window_mode,
        decimate=args.decimate,
        decimate_threshold=args.decimate_threshold,
        site=args.site,
        max_deg=args.max_deg
    )
    result = reduction.run(tasks)
    print(f'Completed tasks: {", ".join(tasks)}')
    print(f'Output files: {", ".join(result["output_files"])}')
    return 0

if __name__ == '__main__':
    sys.exit(main())