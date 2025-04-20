############################################################
# Utilities for interpolating data                         #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np
import pandas as pd

from scipy.spatial import Delaunay
from scipy.interpolate import (
    LinearNDInterpolator, 
    NearestNDInterpolator, 
    CloughTocher2DInterpolator, 
    Rbf
)
from typing import Union

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPR_RBF, ConstantKernel as GPR_Constant
def clean_data(df, key='Dg') -> pd.DataFrame:
    '''
    Clean the input DataFrame (dropping NaNs and averaging duplicates).
    
    Parameters
    ----------
    df        : the input DataFrame
        
    
    Returns
    -------
    df_clean  : the cleaned DataFrame
    '''
    
    df_clean = df.dropna(subset=[key]).copy()
    df_clean = df_clean.groupby(['lon', 'lat'], as_index=False)[key].mean()
    return df_clean[['lon', 'lat', key]]

class Interpolators:
    '''
    Interpolators class for gridding scattered data using various methods.
    
    Notes
    -----
    The class uses nearest interpolation to extrapolate values outside the convex hull of the data.
    '''
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        grid_extent: tuple[float, float, float, float],
        resolution: float, 
        method: str = None,
        resolution_unit: str ='minutes',
        data_key: str = 'Dg',
        verbose: bool = False
    ) -> None:
        '''
        Initialize the Interpolators class.
        
        Parameters
        ----------
        df             : the input DataFrame
        grid_extent    : the extent of the grid (lon_min, lon_max, lat_min, lat_max)
        resolution     : the resolution of the grid (in degrees or minutes)
        method         : the interpolation method to use 
                            'linear'    : linear interpolation
                            'spline'    : cubic spline-like interpolation (Clough-Tocher)
                            'kriging'   : ordinary kriging
                            'rbf'       : radial basis function interpolation
                            'idw'       : inverse distance weighting
                            'biharmonic': biharmonic spline interpolation
                            'gpr'       : Gaussian process regression
        resolution_unit: unit of resolution ('degrees' or 'minutes')
        data_key       : the column name of the data to interpolate (default: 'Dg')
        verbose        : if True, print additional information
        
        Returns
        -------
        None
        '''
        self.df = dataframe
        self.method = method
        self.grid_extent = grid_extent
        self.resolution = resolution
        self.resolution_unit = resolution_unit
        self.data_key = data_key

        if self.method is None:
            print('No interpolation method specified. Defaulting to kriging.') if verbose else None
            self.method = 'kriging'
        
        # Clean data
        self.df_clean = clean_data(self.df, key=self.data_key)
        
        # Convert resolution to degrees if in minutes
        if self.resolution_unit == 'minutes':
            self.resolution = self.resolution / 60.0
        elif self.resolution_unit == 'seconds':
            self.resolution = self.resolution / 3600.0
        elif self.resolution_unit == 'degrees':
            pass
        else:
            raise ValueError('resolution_unit must be \'degrees\', \'minutes\', or \'seconds\'')

        # Create a grid for interpolation
        lon_min, lon_max, lat_min, lat_max = self.grid_extent
        num_x_points = int((lon_max - lon_min) / self.resolution) + 1
        num_y_points = int((lat_max - lat_min) / self.resolution) + 1
        self.lon_grid = np.linspace(lon_min, lon_max, num_x_points)
        self.lat_grid = np.linspace(lat_min, lat_max, num_y_points)

        # Create a meshgrid for interpolation
        self.Lon, self.Lat = np.meshgrid(self.lon_grid, self.lat_grid)
        
        # Create Delaunay triangulation
        self.points = self.df_clean[['lon', 'lat']].values
        self.tri = Delaunay(self.points, qhull_options='Qt Qbb Qc Qz')
        self.values = self.df_clean[self.data_key].values
        
        # if self.method == 'linear' or self.method == 'spline':
        self.neighbor_interp = NearestNDInterpolator(self.points, self.values)

    def scatteredInterpolant(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using Delaunay triangulation and linear interpolation.
        
        Returns
        -------
        grid           : the interpolated grid
        Lon            : 2D array of longitude coordinates
        Lat            : 2D array of latitude coordinates
        '''
        interpolator = LinearNDInterpolator(self.tri, self.values)
        data_linear  = interpolator(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
        data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
        data_interp  = np.where(np.isnan(data_linear), data_nearest, data_linear)
        
        return self.Lon, self.Lat, data_interp
    
    def splineInterpolant(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using cubic spline-like interpolation (Clough-Tocher).
        
        Returns
        -------
        grid           : the interpolated grid
        Lon            : 2D array of longitude coordinates
        Lat            : 2D array of latitude coordinates
        '''
        interpolator = CloughTocher2DInterpolator(self.tri, self.values, fill_value=np.nan)
        data_cubic   = interpolator(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
        data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
        data_interp  = np.where(np.isnan(data_cubic), data_nearest, data_cubic)
        
        return self.Lon, self.Lat, data_interp
    
    def krigingInterpolant(
        self, 
        fall_back_on_error: bool = False, 
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using Ordinary Kriging.
        
        Parameters
        ----------
        fall_back_on_error : if True, fall back to default kriging parameters on error
        kwargs             : additional parameters for OrdinaryKriging
        
        Returns
        -------
        Lon            : 2D array of longitude coordinates
        Lat            : 2D array of latitude coordinates
        data_interp    : 2D array of interpolated values (kriging with nearest neighbor extrapolation)
        zz             : 2D array of raw kriging interpolated values
        ss             : 2D array of kriging variance
        '''
        from pykrige.ok import OrdinaryKriging
        
        # Default parameters for OrdinaryKriging
        default_kriging_params = {
            'variogram_model': 'spherical',
            'nlags': 6,
            'verbose': False,
            'enable_plotting': False
        }

        # Update defaults with user-provided kwargs
        kriging_params = default_kriging_params.copy()
        kriging_params.update(kwargs)
        
        lon = self.df_clean['lon'].values
        lat = self.df_clean['lat'].values
        
        # Initialize Ordinary Kriging
        try:
            ok = OrdinaryKriging(
                x=lon,
                y=lat,
                z=self.values,
                **kriging_params
            )
        except ValueError as e:
            if fall_back_on_error:
                print(f'Warning: Invalid kriging parameter: {str(e)}. Falling back to default parameters.')
                print('See PyKrige documentation: https://pykrige.readthedocs.io/')
                kriging_params = default_kriging_params.copy()
                ok = OrdinaryKriging(
                    x=lon,
                    y=lat,
                    z=self.values,
                    **default_kriging_params
                )
            else:
                import inspect                
                signature = inspect.signature(OrdinaryKriging.__init__)
                for param_name, param in signature.parameters.items():
                    if param_name != 'self':
                        default = param.default if param.default is not inspect.Parameter.empty else "Required"
                        print(f"  {param_name:<23}: Default = {default}")
                
                raise ValueError(
                    f'Invalid kriging parameter: {str(e)}.'
                    'Check kwargs against the valid parameters printed above or see https://pykrige.readthedocs.io/"'
                )
            
        zz, ss       = ok.execute('grid', self.lon_grid, self.lat_grid)
        z_nearest    = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
        data_interp  = np.where(np.isnan(zz), z_nearest, zz)
        # data_interp  = data_interp.reshape(self.Lon.shape)
        
        return self.Lon, self.Lat, data_interp, zz, ss
    
    def rbfInterpolant(
        self, 
        function: str = 'linear', 
        epsilon: float = None,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Radial Basis Function (RBF).
        
        Parameters
        ----------
        function : RBF function type (default 'linear')
                    'linear'      : for linear interpolation
                    'cubic'       : for cubic interpolation
                    'quintic'     : for quintic interpolation
                    'thin_plate'  : for thin plate spline interpolation
                    'gaussian'    : for Gaussian interpolation
                    'inverse'     : for inverse distance weighting
                    'multiquadric': for multiquadric interpolation
        epsilon  : RBF parameter (default None)
        kwargs   : Additional arguments for scipy.interpolate.Rbf
        
        Returns
        -------
        Lon      : 2D array of longitude coordinates
        Lat      : 2D array of latitude coordinates
        data_rbf : 2D array of interpolated values
        '''
        if epsilon is None:
            # Estimate average spacing between points
            from scipy.spatial.distance import pdist
            dists = pdist(self.points)
            epsilon = np.median(dists)

        rbf = Rbf(self.points[:,0], self.points[:,1], self.values, function=function, epsilon=epsilon, **kwargs)
        data_rbf = rbf(self.Lon, self.Lat)
        return self.Lon, self.Lat, data_rbf

    def idwInterpolant(self, power: float = 2.0, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Inverse Distance Weighting (IDW).
        
        Parameters
        ----------
        power     : Power parameter for IDW (default 2)
        eps       : Small value to avoid division by zero
        
        Returns
        -------
        Lon       : 2D array of longitude coordinates
        Lat       : 2D array of latitude coordinates
        zi        : 2D array of interpolated values
        '''
        xi = np.column_stack((self.Lon.ravel(), self.Lat.ravel()))
        x = self.points
        z = self.values
        dist = np.sqrt(((xi[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
        weights = 1.0 / (dist ** power + eps)
        weights /= weights.sum(axis=1, keepdims=True)
        zi = (weights * z).sum(axis=1)
        return self.Lon, self.Lat, zi.reshape(self.Lon.shape)

    def biharmonicSplineInterpolant(
        self, 
        function: str = 'thin_plate',
        epsilon: float = None,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Biharmonic Spline (Thin Plate Spline).
        
        Parameters
        ----------
        kwargs          : Additional arguments for scipy.interpolate.Rbf
        kernel          : Kernel function (default None)
        alpha           : Regularization parameter (default 1e-6)
        **kwargs        : Additional arguments for scipy.interpolate.Rbf
        
        Returns
        -------
        Lon             : 2D array of longitude coordinates
        Lat             : 2D array of latitude coordinates
        data_biharmonic : 2D array of interpolated values
        '''
        if epsilon is None:
            from scipy.spatial.distance import pdist
            dists = pdist(self.points)
            epsilon = np.median(dists)
        
        rbf = Rbf(self.points[:,0], self.points[:,1], self.values, function=function, epsilon=epsilon, **kwargs)
        data_biharmonic = rbf(self.Lon, self.Lat)
        return self.Lon, self.Lat, data_biharmonic

    def gprInterpolant(
        self, 
        kernel=None, 
        alpha=1e-10, 
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Gaussian Process Regression (GPR).
        Parameters
        ----------
        kernel : sklearn.gaussian_process.kernels.Kernel instance (optional)
        alpha  : Value added to the diagonal of the kernel matrix during fitting
        kwargs : Additional arguments for GaussianProcessRegressor
        '''
        if kernel is None:
            from scipy.spatial.distance import pdist
            dists = pdist(self.points)
            median_dist = np.median(dists)
        
            if kernel is None:
                kernel = GPR_Constant(1.0, (1e-6, 1e6)) * GPR_RBF(
                    length_scale=median_dist, 
                    length_scale_bounds=(median_dist/1e6, median_dist*1e6)
                )
        
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, **kwargs)
        gpr.fit(self.points, self.values)
        zi, _ = gpr.predict(np.column_stack((self.Lon.ravel(), self.Lat.ravel())), return_std=True)
        return self.Lon, self.Lat, zi.reshape(self.Lon.shape)

    def run(
        self, 
        method: str = None, 
        **kwargs
    ) -> Union[
        tuple[np.ndarray, np.ndarray, np.ndarray],  # linear/spline/rbf/idw/biharmonic/gpr
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # kriging
    ]:
        '''
        Run the selected interpolation method.

        Parameters
        ----------
        method    : Interpolation method to use ('linear', 'spline', 'kriging').
                    If not provided, uses self.method if it exists.
        kwargs    : additional keyword arguments for krigingInterpolant method.

        Returns
        -------
        Interpolated results as returned by the selected method.
        '''
        method = method or getattr(self, 'method', None)
        if method is None:
            raise ValueError('No interpolation method specified. Provide \'method\' argument or set \'self.method\'.')

        if method == 'linear':
            return self.scatteredInterpolant()
        elif method == 'spline':
            return self.splineInterpolant()
        elif method == 'kriging':
            return self.krigingInterpolant(**kwargs)
        elif method == 'rbf':
            return self.rbfInterpolant(**kwargs)
        elif method == 'idw':
            return self.idwInterpolant(**kwargs)
        elif method == 'biharmonic':
            return self.biharmonicSplineInterpolant(**kwargs)
        elif method == 'gpr':
            return self.gprInterpolant(**kwargs)
        else:
            raise ValueError(f'Unknown interpolation method: {method}')