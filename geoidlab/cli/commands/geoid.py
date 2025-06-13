############################################################
# Geoid  CLI interface                                     #
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

from geoidlab.cli.commands.reference import add_reference_arguments, GGMSynthesis
from geoidlab.cli.commands.reduce import add_reduce_arguments, GravityReduction
from geoidlab.cli.commands.topo import add_topo_arguments, TopographicQuantities
from geoidlab.cli.commands.utils.common import directory_setup, to_seconds
from geoidlab.geoid import ResidualGeoid
from geoidlab.icgem import get_ggm_tide_system
from geoidlab.utils.io import save_to_netcdf

METHODS_DICT = {
    'hg': "Heck & Gruninger's modification",
    'wg': "Wong & Gore's modification",
    'og': "Original Stokes'",
    'ml': "Meissl's modification",
}

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

def copy_arguments(src_parser, dst_parser, exclude=None) -> None:
    '''
    Copy arguments from one src_parser to dst_parser, skipping duplicates based on dest
    
    Parameters
    ----------
    src_parser  : Source ArgumentParser with arguments to copy
    dest_parser : Destination ArgumentParser to receive arguments
    exclude     : List of dest names to exclude (optional)
    '''    
    exclude = exclude or []
    existing_dests = {action.dest for action in dst_parser._actions}
    for action in src_parser._actions:
        if action.dest not in existing_dests and action.dest not in exclude:
            kwargs = {
                'dest': action.dest,
                'help': action.help,
                'default': action.default,
            }
            args = action.option_strings if action.option_strings else [action.dest]

            # Handle different action types
            if isinstance(action, argparse._StoreAction):
                kwargs['type'] = action.type
                kwargs['choices'] = action.choices
                kwargs['nargs'] = action.nargs
            elif isinstance(action, argparse._StoreTrueAction):
                kwargs['action'] = 'store_true'
            elif isinstance(action, argparse._StoreFalseAction):
                kwargs['action'] = 'store_false'
            elif isinstance(action, argparse._StoreConstAction):
                kwargs['action'] = 'store_const'
                kwargs['const'] = action.const
            else:
                raise ValueError(f"Unsupported action type for '{action.dest}': {type(action)}")

            dst_parser.add_argument(*args, **kwargs)
            existing_dests.add(action.dest)

def add_geoid_arguments(parser) -> None:
    # parser = argparse.ArgumentParser(description='Geoid computation CLI')
    
    tem_parser_ref = argparse.ArgumentParser(add_help=False)
    add_reference_arguments(tem_parser_ref)
    
    tem_parser_topo = argparse.ArgumentParser(add_help=False)
    add_topo_arguments(tem_parser_topo)
    
    tem_parser_helmert = argparse.ArgumentParser(add_help=False)
    add_reduce_arguments(tem_parser_helmert)
    
    # Arguments to exclude
    exclude_list = ['grid', 'do', 'start', 'end', 'ellipsoidal_correction']
    
    # Merge arguments into the main parser
    copy_arguments(tem_parser_ref, parser, exclude=exclude_list)
    copy_arguments(tem_parser_topo, parser, exclude=exclude_list)
    copy_arguments(tem_parser_helmert, parser, exclude=exclude_list)
    
    # Add marine data arguments
    parser.add_argument('--marine-data', type=str,
                      help='Input file with marine gravity data (lon, lat, Dg, height)')
    parser.add_argument('--decimate', action='store_true',
                      help='Decimate marine data. Default observations to retain is 600. Use --decimate-threshold to change this.')
    parser.add_argument('--decimate-threshold', type=int, default=600,
                      help='Threshold for automatic decimation of marine data (default: 600 points).')
    
    # Add residual computation method argument
    parser.add_argument('--residual-method', type=str, choices=['station', 'gridded'], 
                      default='station',
                      help='Method for computing residual anomalies: station (compute at observation points) or gridded (compute after gridding). Note: The gridded method uses Bouguer anomalies which may lead to larger indirect effects (Torge et al., 2023).')
    
    # Existing geoid arguments
    parser.add_argument('--sph-cap', type=float, default=1.0,
                      help='Spherical cap for integration in degrees (default: 1.0)')
    parser.add_argument('--method', type=str, default='hg', choices=['hg', 'wg', 'ml', 'og'],
                      help='Geoid computation method (default: hg). Options: Heck & Gruninger (hg), Wong & Gore (wg), Meissl (ml), original (og)')
    parser.add_argument('--ind-grid-size', type=float, default=30,
                      help='Grid resolution for computing indirect effect. Keep this in seconds. Default: 30 seconds')
    parser.add_argument('--target-tide-system', type=str, default='tide_free', choices=['mean_tide', 'tide_free', 'zero_tide'],
                      help='The tide system that the final geoid should be in. Default: tide_free')
    
class ResidualAnomalyComputation:
    '''Class to handle computation of residual gravity anomalies using different methods'''
    
    def __init__(
        self,
        input_file: str,
        marine_data: str = None,
        model: str = None,
        model_dir: str | Path = None,
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
        max_deg: int = 90,
        residual_method: str = 'station'
    ) -> None:
        self.input_file = input_file
        self.marine_data = marine_data
        self.model = model
        self.model_dir = Path(model_dir) if model_dir else Path(proj_name) / 'downloads'
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
        self.residual_method = residual_method
        
        directory_setup(proj_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_marine_data(self) -> pd.DataFrame:
        '''Load and process marine gravity data'''
        if not self.marine_data:
            return None
            
        marine_path = Path(self.marine_data)
        if marine_path.suffix == '.csv':
            marine_df = pd.read_csv(marine_path)
        elif marine_path.suffix in ['.xlsx', '.xls']:
            marine_df = pd.read_excel(marine_path)
        elif marine_path.suffix == '.txt':
            marine_df = pd.read_csv(marine_path, delimiter='\t')
        else:
            raise ValueError(f'Unsupported file format: {marine_path.suffix}')
            
        if not all(col in marine_df.columns for col in ['lon', 'lat', 'height', 'Dg']):
            raise ValueError('Marine data must contain columns: lon, lat, height, and Dg')
            
        # Apply decimation if requested
        if self.decimate and len(marine_df) > self.decimate_threshold:
            marine_df = decimate_data(marine_df, n_points=self.decimate_threshold, verbose=True)
            
        return marine_df
    
    def _compute_gravity_anomalies(self, anomaly_type: str) -> pd.DataFrame:
        '''Compute gravity anomalies using GravityReduction
        
        Parameters
        ----------
        anomaly_type : str
            Type of anomaly to compute ('helmert' or 'bouguer')
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the computed anomalies with coordinates and height
        '''
        # First, load the original input data to get height information
        input_df = pd.read_csv(self.input_file)
        if not all(col in input_df.columns for col in ['lon', 'lat', 'height']):
            raise ValueError('Input file must contain columns: lon, lat, height')
        
        # Compute anomalies
        gravity_reduction = GravityReduction(
            input_file=self.input_file,
            model=self.model,
            model_dir=self.model_dir,
            gravity_tide=self.gravity_tide,
            ellipsoid=self.ellipsoid,
            converted=self.converted,
            grid=True,
            grid_size=self.grid_size,
            grid_unit=self.grid_unit,
            grid_method=self.grid_method,
            bbox=self.bbox,
            bbox_offset=self.bbox_offset,
            proj_name=self.proj_name,
            topo=self.topo,
            tc_file=self.tc_file,
            radius=self.radius,
            interp_method=self.interp_method,
            parallel=self.parallel,
            chunk_size=self.chunk_size,
            atm=self.atm,
            atm_method=self.atm_method,
            ellipsoidal_correction=self.ellipsoidal_correction,
            window_mode=self.window_mode,
            site=self.site,
            max_deg=self.max_deg
        )
        
        # Run gravity reduction
        result = gravity_reduction.run([anomaly_type])
        
        # Load the computed anomalies
        anomaly_file = self.output_dir / f'{anomaly_type}.csv'
        if not anomaly_file.exists():
            raise FileNotFoundError(f'Expected anomaly file {anomaly_file} not found after computation')
        
        # Load computed anomalies and merge with height information
        anomaly_df = pd.read_csv(anomaly_file)
        merged_df = pd.merge(
            anomaly_df,
            input_df[['lon', 'lat', 'height']],
            on=['lon', 'lat'],
            how='left'
        )
        
        if merged_df['height'].isna().any():
            print('Warning: Some points in the computed anomalies do not have height information')
        
        return merged_df
    
    def compute_residual_anomalies_station(self) -> tuple[xr.Dataset, str]:
        '''Compute residual anomalies at observation points (Approach 1)
        
        Returns
        -------
        tuple
            (Dg_res, output_file) where Dg_res is an xarray Dataset
            containing the gridded residual anomalies and output_file is the path to the saved file
        '''
        # 1. Compute Helmert anomalies with corrections if requested
        print('Computing Helmert anomalies...')
        surface_df = self._compute_gravity_anomalies('helmert')
        
        # Verify we have all required columns
        required_cols = ['lon', 'lat', 'height', 'helmert']
        if not all(col in surface_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in surface_df.columns]
            raise ValueError(f'Missing required columns in surface data: {missing}')
        
        # 2. Load and process marine data if available
        marine_df = self._load_marine_data()
        
        # 3. Compute GGM anomalies for both surface and marine points
        print('Computing GGM anomalies for surface points...')
        ggm_surface = GGMSynthesis(
            model=self.model,
            max_deg=self.max_deg,
            model_dir=self.model_dir,
            output_dir=self.output_dir,
            ellipsoid=self.ellipsoid,
            chunk_size=self.chunk_size,
            parallel=self.parallel,
            tide_system=self.gravity_tide,
            converted=True,
            bbox=self.bbox,
            bbox_offset=self.bbox_offset,
            grid_size=self.grid_size,
            grid_unit=self.grid_unit,
            proj_name=self.proj_name,
            input_file=str(self.output_dir / 'surface_points.csv')  # Save surface points to a temporary file
        )
        # Save surface points to a temporary file for GGM computation
        surface_df[['lon', 'lat', 'height']].to_csv(ggm_surface.input_file, index=False)
        surface_ggm = ggm_surface.compute_gravity_anomaly()
        surface_df['Dg_ggm'] = pd.read_csv(ggm_surface.output_dir / 'Dg_ggm.csv')['Dg_ggm']
        
        # Compute GGM anomalies for marine points if available
        if marine_df is not None:
            print('Computing GGM anomalies for marine points...')
            ggm_marine = GGMSynthesis(
                model=self.model,
                max_deg=self.max_deg,
                model_dir=self.model_dir,
                output_dir=self.output_dir,
                ellipsoid=self.ellipsoid,
                chunk_size=self.chunk_size,
                parallel=self.parallel,
                tide_system=self.gravity_tide,
                converted=True,
                bbox=self.bbox,
                bbox_offset=self.bbox_offset,
                grid_size=self.grid_size,
                grid_unit=self.grid_unit,
                proj_name=self.proj_name,
                input_file=str(self.output_dir / 'marine_points.csv')  # Save marine points to a temporary file
            )
            # Save marine points to a temporary file for GGM computation
            marine_df[['lon', 'lat', 'height']].to_csv(ggm_marine.input_file, index=False)
            marine_ggm = ggm_marine.compute_gravity_anomaly()
            marine_df['Dg_ggm'] = pd.read_csv(ggm_marine.output_dir / 'Dg_ggm.csv')['Dg_ggm']
        
        # 4. Compute residuals at observation points
        surface_df['Dg_res'] = surface_df['helmert'] - surface_df['Dg_ggm']
        if marine_df is not None:
            marine_df['Dg_res'] = marine_df['Dg'] - marine_df['Dg_ggm']
        
        # 5. Merge surface and marine residuals
        if marine_df is not None:
            combined_df = pd.concat([
                surface_df[['lon', 'lat', 'Dg_res']],
                marine_df[['lon', 'lat', 'Dg_res']]
            ], ignore_index=True)
        else:
            combined_df = surface_df[['lon', 'lat', 'Dg_res']]
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['lon', 'lat'])
        
        # 6. Grid the combined residuals
        from geoidlab.utils.interpolators import Interpolators
        
        grid_size = to_seconds(self.grid_size, self.grid_unit) / 3600.0
        grid_extent = (
            self.bbox[0] - self.bbox_offset,
            self.bbox[1] + self.bbox_offset,
            self.bbox[2] - self.bbox_offset,
            self.bbox[3] + self.bbox_offset
        )
        
        interpolator = Interpolators(
            dataframe=combined_df,
            grid_extent=grid_extent,
            resolution=grid_size,
            resolution_unit='degrees',
            data_key='Dg_res',
            verbose=False
        )
        
        if self.grid_method == 'kriging':
            lon, lat, gridded_values, *_ = interpolator.run(method=self.grid_method, merge=True)
        elif self.grid_method == 'lsc':
            lon, lat, gridded_values, *_ = interpolator.run(method=self.grid_method, merge=True, robust_covariance=True, covariance_model='exp')
        elif self.grid_method == 'rbf':
            lon, lat, gridded_values = interpolator.run(method='rbf', function='linear', merge=False)
        else:
            lon, lat, gridded_values = interpolator.run(method=self.grid_method, merge=True)
        
        # Create dataset
        residual_ds = xr.Dataset(
            data_vars={
                'Dg_res': xr.DataArray(
                    data=gridded_values,
                    dims=['lat', 'lon'],
                    attrs={
                        'units': 'mGal',
                        'long_name': 'Residual Gravity Anomaly',
                        'description': 'Residual gravity anomaly computed at observation points'
                    }
                )
            },
            coords={
                'lat': lat[:, 0],
                'lon': lon[0, :]
            }
        )
        
        # Save to NetCDF
        output_file = self.output_dir / 'Dg_res.nc'
        residual_ds.to_netcdf(output_file, mode='w')
        
        return residual_ds, str(output_file)
    
    def compute_residual_anomalies_gridded(self) -> tuple[xr.Dataset, str]:
        '''Compute residual anomalies using gridded approach (Approach 2)
        
        Returns
        -------
        tuple
            (Dg_res, output_file) where Dg_res is an xarray Dataset
            containing the gridded residual anomalies and output_file is the path to the saved file
        '''
        # 1. Compute Bouguer anomalies
        print('Computing Bouguer anomalies...')
        bouguer_df = self._compute_gravity_anomalies('bouguer')
        
        # 2. Load and process marine data if available
        marine_df = self._load_marine_data()
        
        # 3. Merge surface and marine Bouguer anomalies
        if marine_df is not None:
            # Convert marine free-air anomalies to Bouguer anomalies
            marine_df['bouguer'] = marine_df['Dg'] - 0.1119 * marine_df['height']
            # marine_df['bouguer'] = marine_df['Dg']
            combined_df = pd.concat([
                bouguer_df[['lon', 'lat', 'bouguer']],
                marine_df[['lon', 'lat', 'bouguer']]
            ], ignore_index=True)
        else:
            combined_df = bouguer_df[['lon', 'lat', 'bouguer']]
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['lon', 'lat'])
        
        # 4. Grid the combined Bouguer anomalies
        from geoidlab.utils.interpolators import Interpolators
        
        grid_size = to_seconds(self.grid_size, self.grid_unit) / 3600.0
        grid_extent = (
            self.bbox[0] - self.bbox_offset,
            self.bbox[1] + self.bbox_offset,
            self.bbox[2] - self.bbox_offset,
            self.bbox[3] + self.bbox_offset
        )
        
        interpolator = Interpolators(
            dataframe=combined_df,
            grid_extent=grid_extent,
            resolution=grid_size,
            resolution_unit='degrees',
            data_key='bouguer',
            verbose=False
        )
        
        if self.grid_method == 'kriging':
            lon, lat, gridded_values, *_ = interpolator.run(method=self.grid_method, merge=True)
        elif self.grid_method == 'lsc':
            lon, lat, gridded_values, *_ = interpolator.run(method=self.grid_method, merge=True, robust_covariance=True, covariance_model='exp')
        elif self.grid_method == 'rbf':
            lon, lat, gridded_values = interpolator.run(method='rbf', function='linear', merge=False)
        else:
            lon, lat, gridded_values = interpolator.run(method=self.grid_method, merge=True)
        
        # Create initial dataset with gridded Bouguer anomalies
        gridded_ds = xr.Dataset(
            data_vars={
                'bouguer': xr.DataArray(
                    data=gridded_values,
                    dims=['lat', 'lon'],
                    attrs={
                        'units': 'mGal',
                        'long_name': 'Bouguer Gravity Anomaly',
                        'description': 'Gridded Bouguer gravity anomaly'
                    }
                )
            },
            coords={
                'lat': lat[:, 0],
                'lon': lon[0, :]
            }
        )
        
        # 5. Compute terrain correction and other corrections on the same grid
        if self.topo or self.tc_file:
            print('Computing terrain correction on the same grid...')
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
                grid_size=self.grid_size,
                window_mode=self.window_mode,
                parallel=self.parallel,
                interp_method=self.interp_method
            )
            topo_workflow._initialize_terrain()
            tc_result = topo_workflow.run(['terrain-correction'])
            tc_grid = xr.open_dataset(tc_result['output_files'][0])
            gridded_ds['tc'] = tc_grid['tc']
        
        # Compute other corrections if requested
        if any([self.site, self.ellipsoidal_correction, self.atm]):
            print('Computing additional corrections on the same grid...')
            if self.site:
                site_result = topo_workflow.run(['site'])
                site_grid = xr.open_dataset(site_result['output_files'][0])
                gridded_ds['site'] = site_grid['Dg_SITE']
            
            if self.ellipsoidal_correction:
                ggm = GGMSynthesis(
                    model=self.model,
                    max_deg=self.max_deg,
                    model_dir=self.model_dir,
                    output_dir=self.output_dir,
                    ellipsoid=self.ellipsoid,
                    chunk_size=self.chunk_size,
                    parallel=self.parallel,
                    tide_system=self.gravity_tide,
                    converted=True,
                    bbox=self.bbox,
                    bbox_offset=self.bbox_offset,
                    grid_size=self.grid_size,
                    grid_unit=self.grid_unit,
                    proj_name=self.proj_name
                )
                ec_grid = ggm.compute_ellipsoidal_correction()
                gridded_ds['ellipsoidal_correction'] = ec_grid['Dg_ELL']
            
            if self.atm:
                atm_result = topo_workflow.run(['atm-corr'])
                atm_grid = xr.open_dataset(atm_result['output_files'][0])
                gridded_ds['atm_correction'] = atm_grid['Dg_atm']
        
        # 6. Compute GGM anomalies on the same grid
        print('Computing GGM anomalies on the same grid...')
        ggm = GGMSynthesis(
            model=self.model,
            max_deg=self.max_deg,
            model_dir=self.model_dir,
            output_dir=self.output_dir,
            ellipsoid=self.ellipsoid,
            chunk_size=self.chunk_size,
            parallel=self.parallel,
            tide_system=self.gravity_tide,
            converted=True,
            bbox=self.bbox,
            bbox_offset=self.bbox_offset,
            grid_size=self.grid_size,
            grid_unit=self.grid_unit,
            proj_name=self.proj_name
        )
        ggm_result = ggm.run(['gravity-anomaly'])
        ggm_grid = xr.open_dataset(ggm_result['output_files'][0])
        gridded_ds['Dg_ggm'] = ggm_grid['Dg']
        
        # 7. Compute residuals from gridded data
        # Start with Bouguer anomalies
        residual = gridded_ds['bouguer'].copy()
        
        # Add terrain correction if computed
        if 'tc' in gridded_ds:
            residual = residual + gridded_ds['tc']
        
        # Add other corrections if computed
        for correction in ['site', 'ellipsoidal_correction', 'atm_correction']:
            if correction in gridded_ds:
                residual = residual + gridded_ds[correction]
        
        # Subtract GGM anomalies
        residual = residual - gridded_ds['Dg_ggm']
        
        # Create final dataset
        residual_ds = xr.Dataset(
            data_vars={
                'Dg_res': xr.DataArray(
                    data=residual.values,
                    dims=['lat', 'lon'],
                    attrs={
                        'units': 'mGal',
                        'long_name': 'Residual Gravity Anomaly',
                        'description': 'Residual gravity anomaly computed from gridded Bouguer anomalies'
                    }
                )
            },
            coords=gridded_ds.coords
        )
        
        # Save to NetCDF
        output_file = self.output_dir / 'Dg_res.nc'
        residual_ds.to_netcdf(output_file, mode='w')
        
        return residual_ds, str(output_file)

def main(args=None) -> None:
    '''
    Main function to handle command line arguments and execute geoid computation.
    '''
    
    if args is None:
        parser = argparse.ArgumentParser(
            description=(
                'Complete workflow for geoid computation using the remove-compute-restore (RCR) method.'
                'Options for solving Stokes\'s integral include Heck & Gruninger (hg), Wong & Gore (wg), Meissl (ml), and original (og).'
            )
        )
        add_geoid_arguments(parser)
        args = parser.parse_args()
    
    # Set up directories
    directory_setup(args.proj_name)
    model_dir = Path(args.proj_name) / 'downloads'
    output_dir = Path(args.proj_name) / 'results'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate and define computation grid
    if not (args.bbox and args.grid_size and args.grid_unit):
        raise ValueError('bbox, grid_size, and grid_unit must be provided for geoid computation')
    bbox = args.bbox
    grid_size = args.grid_size
    grid_unit = args.grid_unit
    
    model_path = (Path(model_dir) / Path(args.model)).with_suffix('.gfc')
    ggm_tide = get_ggm_tide_system(icgem_file=model_path, model_dir=model_dir)
    
    # Initialize residual computation
    residual_computation = ResidualAnomalyComputation(
        input_file=args.input_file,
        marine_data=args.marine_data,
        model=args.model,
        model_dir=model_dir,
        gravity_tide=args.gravity_tide,
        ellipsoid=args.ellipsoid,
        converted=args.converted,
        grid=True,
        grid_size=grid_size,
        grid_unit=grid_unit,
        grid_method=args.grid_method,
        bbox=bbox,
        bbox_offset=args.bbox_offset,
        proj_name=args.proj_name,
        topo=args.topo,
        tc_file=getattr(args, 'tc_file', None),
        radius=args.radius,
        interp_method=args.interp_method,
        parallel=args.parallel,
        chunk_size=args.chunk_size,
        atm=args.atm,
        atm_method=args.atm_method,
        ellipsoidal_correction=args.ellipsoidal_correction,
        window_mode=args.window_mode,
        decimate=args.decimate,
        decimate_threshold=args.decimate_threshold,
        site=args.site,
        max_deg=args.max_deg,
        residual_method=args.residual_method
    )
    
    # Compute residual anomalies using the chosen method
    if args.residual_method == 'station':
        print('Computing residual anomalies at observation points...')
        Dg_res_ds, Dg_res_file = residual_computation.compute_residual_anomalies_station()
    else:
        print('Computing residual anomalies using gridded approach...')
        Dg_res_ds, Dg_res_file = residual_computation.compute_residual_anomalies_gridded()
    
    print(f'Computed residual anomalies saved to {Dg_res_file}')
    
    # Step 3: Calculate the residual geoid (N_res)
    print('Computing the residual geoid as the sum of the inner and outer zones contributions...')
    sub_grid = bbox  # Use the same grid as input for simplicity

    residual_geoid = ResidualGeoid(
        res_anomaly=Dg_res_ds.rename({'Dg_res': 'Dg'}),
        sph_cap=args.sph_cap,
        sub_grid=sub_grid,
        method=args.method,
        ellipsoid=args.ellipsoid,
        nmax=args.max_deg,
        window_mode=args.window_mode
    )
    N_res = residual_geoid.compute_geoid()
    print(f'Saving residual geoid to {output_dir}/N_res.nc')
    save_to_netcdf(
        data=N_res,
        lon=residual_geoid.res_anomaly_P['lon'].values,
        lat=residual_geoid.res_anomaly_P['lat'].values,
        dataset_key='N_res',
        filepath=output_dir / 'N_res.nc',
        tide_system=ggm_tide,
        method=METHODS_DICT[args.method],
    )
    
    print('Residual geoid computation completed.\n')
    
    # Step 4: Calculate reference geoid (N_ggm)
    ggm_synthesis = GGMSynthesis(
        model=args.model,
        max_deg=args.max_deg,
        model_dir=model_dir,
        output_dir=output_dir,
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        parallel=args.parallel,
        tide_system=args.gravity_tide,
        converted=True,
        bbox=bbox,
        grid_size=grid_size,
        grid_unit=grid_unit,
        proj_name=args.proj_name,
        icgem=args.icgem,
        dtm_model= args.dtm_model
    )
    
    ggm_result = ggm_synthesis.run(['reference-geoid'])
    N_ggm_file = ggm_result['output_files'][0]
    N_ggm_ds = xr.open_dataset(N_ggm_file)
    N_ggm = N_ggm_ds['N_ref']
    print(f'Computed reference geoid saved to {N_ggm_file}')
    
    # Step 5: Calculate indirect effect (N_ind)
    topo_quantities = TopographicQuantities(
        topo=args.topo,
        ref_topo=args.ref_topo,
        model_dir=model_dir,
        output_dir=output_dir,
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        radius=args.radius,
        parallel=args.parallel,
        bbox=bbox,
        bbox_offset=args.bbox_offset,
        grid_size=args.ind_grid_size,
        proj_name=args.proj_name,
        window_mode=args.window_mode,
        interp_method=args.interp_method,
    )
    topo_result = topo_quantities.run(['indirect-effect'])
    N_ind_file = topo_result['output_files'][0]
    N_ind_ds = xr.open_dataset(N_ind_file)
    N_ind = N_ind_ds['N_ind']
    print(f'Computed indirect effect saved to {N_ind_file}')
    
    # Step 6: Calculate total geoid (N = N_ggm + N_res + N_ind)
    if args.ind_grid_size != args.grid_size:
        print('Resampling indirect effect to the same grid as the reference and residual geoids...')
        N_ind = N_ind.interp(lon=N_ggm_ds['lon'], lat=N_ggm_ds['lat'], method='linear')
    
    print('Calculating total geoid as the sum of reference, residual, and indirect effects...')
    N = N_ggm.values + N_res + N_ind.values
    
    output_file = output_dir / 'N.nc'
    
    # Convert tide system if needed
    if args.target_tide_system != ggm_tide:
        from geoidlab.tide import GeoidTideSystemConverter
        import numpy as np
        print(f'Converting geoid from {ggm_tide} to {args.target_tide_system} tide system...')
        phi, _ = np.meshgrid(N_ggm_ds['lat'], N_ggm_ds['lon'], indexing='ij')
        converter = GeoidTideSystemConverter(phi=phi, geoid=N)
        conversion_map = {
            ('mean_tide', 'tide_free'): 'mean2free',
            ('tide_free', 'mean_tide'): 'free2mean',
            ('mean_tide', 'zero_tide'): 'mean2zero',
            ('zero_tide', 'mean_tide'): 'zero2mean',
            ('zero_tide', 'tide_free'): 'zero2free',
            ('tide_free', 'zero_tide'): 'free2zero'
        }
        conversion_key = (ggm_tide, args.target_tide_system)
        if conversion_key not in conversion_map:
            raise ValueError(f'No conversion defined from {ggm_tide} to {args.target_tide_system}')
        
        # Perform conversion
        conversion_method = getattr(converter, conversion_map[conversion_key])
        N = conversion_method()
        print(f'Geoid converted to {args.target_tide_system} tide system.')
    
    print(f'Writing geoid to {output_file}')
    save_to_netcdf(
        data=N,
        lon=N_ggm_ds['lon'].values,
        lat=N_ggm_ds['lat'].values,
        dataset_key='N',
        filepath=output_file,
        tide_system=args.target_tide_system if args.target_tide_system else ggm_tide,
        method=METHODS_DICT[args.method],
    )
    
    print(f'Geoid heights written to {output_file}.\n')
    print('Geoid computation completed successfully.\n\n\n')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())