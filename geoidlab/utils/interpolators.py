############################################################
# Utilities for interpolating data                         #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np
import pandas as pd

from geoidlab.utils.lsc_helper_funcs import (
    fit_exponential_covariance,
    fit_gaussian_covariance,
    lsc_exponential,
    lsc_gaussian
)

from scipy.spatial import Delaunay
from scipy.interpolate import (
    LinearNDInterpolator, 
    NearestNDInterpolator, 
    CloughTocher2DInterpolator, 
    Rbf
)

from typing import Union, Tuple, Optional


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
                            'lsc'       : Least Squares Collocation
        resolution_unit: unit of resolution ('degrees' or 'minutes' or 'seconds')
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
        self.verbose = verbose

        if self.method is None:
            print('No interpolation method specified. Defaulting to kriging unless later specified in `run` method.') if verbose else None
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

    def scatteredInterpolant(self, merge: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using Delaunay triangulation and linear interpolation.
        
        Parameters
        ----------
        merge           : If True, merge results with nearest neighbor extrapolation
        
        Returns
        -------
        grid           : the interpolated grid
        Lon            : 2D array of longitude coordinates
        Lat            : 2D array of latitude coordinates
        '''
        interpolator = LinearNDInterpolator(self.tri, self.values)
        data_linear  = interpolator(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)

        if merge:
            data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            data_linear  = np.where(np.isnan(data_linear), data_nearest, data_linear)
        
        return self.Lon, self.Lat, data_linear

    def splineInterpolant(self, merge: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using cubic spline-like interpolation (Clough-Tocher).
        
        Parameters
        ----------
        merge           : If True, merge results with nearest neighbor extrapolation
        
        Returns
        -------
        grid           : the interpolated grid
        Lon            : 2D array of longitude coordinates
        Lat            : 2D array of latitude coordinates
        '''
        interpolator = CloughTocher2DInterpolator(self.tri, self.values, fill_value=np.nan)
        data_cubic   = interpolator(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)

        if merge:
            data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            data_cubic   = np.where(np.isnan(data_cubic), data_nearest, data_cubic)
        
        return self.Lon, self.Lat, data_cubic

    def krigingInterpolant(
        self, 
        fall_back_on_error: bool = False, 
        merge: bool = True,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using Ordinary Kriging.
        
        Parameters
        ----------
        fall_back_on_error : if True, fall back to default kriging parameters on error
        merge             : If True, merge results with nearest neighbor extrapolation
        kwargs            : additional parameters for OrdinaryKriging
        
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
                        default = param.default if param.default is not inspect.Parameter.empty else 'Required'
                        print(f'  {param_name:<23}: Default = {default}')
                
                raise ValueError(
                    f'Invalid kriging parameter: {str(e)}.'
                    'Check kwargs against the valid parameters printed above or see https://pykrige.readthedocs.io/'
                )
            
        zz, ss       = ok.execute('grid', self.lon_grid, self.lat_grid)

        if merge:
            z_nearest    = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            zz           = np.where(np.isnan(zz), z_nearest, zz)
        
        return self.Lon, self.Lat, zz, zz, ss

    def rbfInterpolant(
        self, 
        function: str = 'linear', 
        epsilon: float = None,
        merge: bool = True,
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
        merge    : If True, merge results with nearest neighbor extrapolation
        kwargs   : Additional arguments for scipy.interpolate.Rbf
        
        Returns
        -------
        Lon      : 2D array of longitude coordinates
        Lat      : 2D array of latitude coordinates
        data_rbf : 2D array of interpolated values
        '''
        valid_funcs = ['linear', 'cubic', 'quintic', 'thin_plate', 'gaussian', 'inverse', 'multiquadric']
        if function not in valid_funcs:
            raise ValueError(f'Invalid function type: {function}. Valid functions: {valid_funcs}.')
        
        if epsilon is None:
            # Estimate average spacing between points
            from scipy.spatial.distance import pdist
            dists = pdist(self.points)
            epsilon = np.median(dists)

        rbf = Rbf(self.points[:,0], self.points[:,1], self.values, function=function, epsilon=epsilon, **kwargs)
        data_rbf = rbf(self.Lon, self.Lat)

        if merge:
            from scipy.spatial import Delaunay
            hull = Delaunay(self.points)
            mask = hull.find_simplex(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))) >= 0
            mask = mask.reshape(self.Lon.shape)

            # Blend with nearest neighbor extrapolation
            data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            data_rbf = np.where(mask, data_rbf, data_nearest)

        return self.Lon, self.Lat, data_rbf

    def idwInterpolant(self, power: float = 2.0, eps: float = 1e-12, merge: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Inverse Distance Weighting (IDW).
        
        Parameters
        ----------
        power     : Power parameter for IDW (default 2)
        eps       : Small value to avoid division by zero
        merge     : If True, merge results with nearest neighbor extrapolation
        
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
        zi = (weights * z).sum(axis=1).reshape(self.Lon.shape)

        if merge:
            from scipy.spatial import Delaunay
            hull = Delaunay(self.points)
            mask = hull.find_simplex(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))) >= 0
            mask = mask.reshape(self.Lon.shape)

            # Blend with nearest neighbor extrapolation
            data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            zi = np.where(mask, zi, data_nearest)

        return self.Lon, self.Lat, zi

    def biharmonicSplineInterpolant(
        self, 
        function: str = 'thin_plate',
        epsilon: float = None,
        merge: bool = True,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Biharmonic Spline (Thin Plate Spline).
        
        Parameters
        ----------
        kwargs          : Additional arguments for scipy.interpolate.Rbf
        kernel          : Kernel function (default None)
        alpha           : Regularization parameter (default 1e-6)
        merge           : If True, merge results with nearest neighbor extrapolation
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

        if merge:
            from scipy.spatial import Delaunay
            hull = Delaunay(self.points)
            mask = hull.find_simplex(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))) >= 0
            mask = mask.reshape(self.Lon.shape)

            # Blend with nearest neighbor extrapolation
            data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            data_biharmonic = np.where(mask, data_biharmonic, data_nearest)

        return self.Lon, self.Lat, data_biharmonic

    def gprInterpolant(
        self, 
        kernel=None, 
        alpha=1e-2, 
        normalize_y=True, 
        merge: bool = True,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate using Gaussian Process Regression (GPR).
        
        Parameters
        ----------
        kernel      : sklearn.gaussian_process.kernels.Kernel instance (optional)
        alpha       : Regularization parameter (default 1e-2)
        normalize_y : Whether to normalize the target values (default True)
        merge       : If True, merge results with nearest neighbor extrapolation
        kwargs      : Additional arguments for GaussianProcessRegressor
        
        Returns
        -------
        Lon         : 2D array of longitude coordinates
        Lat         : 2D array of latitude coordinates
        predictions : 2D array of interpolated values
        '''
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        from scipy.spatial.distance import pdist

        # Estimate median distance for kernel initialization
        median_dist = np.median(pdist(self.points))
        if kernel is None:
            kernel = ConstantKernel(
                1.0, (1e-3, 1e3)) * RBF(
                    length_scale=median_dist, 
                    length_scale_bounds=(median_dist/100, median_dist*10
                )
            )

        # Initialize and fit GPR
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, **kwargs)
        gpr.fit(self.points, self.values)

        # Predict
        predictions, _ = gpr.predict(np.column_stack((self.Lon.ravel(), self.Lat.ravel())), return_std=True)
        predictions = predictions.reshape(self.Lon.shape)

        if merge:
            from scipy.spatial import Delaunay
            hull = Delaunay(self.points)
            mask = hull.find_simplex(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))) >= 0
            mask = mask.reshape(self.Lon.shape)

            # Blend with nearest neighbor extrapolation
            data_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel()))).reshape(self.Lon.shape)
            predictions = np.where(mask, predictions, data_nearest)

        return self.Lon, self.Lat, predictions

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

        method_map = {
            'linear'    : self.scatteredInterpolant,
            'spline'    : self.splineInterpolant,
            'kriging'   : self.krigingInterpolant,
            'kriging_chunked': self.krigingInterpolant_chunked,
            'rbf'       : self.rbfInterpolant,
            'idw'       : self.idwInterpolant,
            'biharmonic': self.biharmonicSplineInterpolant,
            'gpr'       : self.gprInterpolant,
            'lsc'       : self.lscInterpolant
        }
        
        if method not in method_map:
            raise ValueError(f'Unknown interpolation method: {method}')
        
        return method_map[method](**kwargs)
    
    def lscInterpolant(
        self, 
        N=None, 
        robust_covariance=False, 
        n_jobs: int = -1,
        chunk_size: Optional[int] = 1000,
        cache_dir: Optional[str] = None,
        use_chunking: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        '''
        Least Squares Collocation interpolation with parallel processing and memory optimization.
        
        Parameters
        ----------
        N                 : Noise variance array (default: 1e-6 for all points)
        robust_covariance : Use robust (median-based) empirical covariance estimation
        n_jobs           : Number of parallel jobs (-1 for all cores)
        chunk_size       : Size of chunks for parallel processing (default: 1000)
        cache_dir        : Directory to store cached computations (None for no caching)
        use_chunking     : Whether to process data in chunks (default: True)
        **kwargs         : Additional parameters including:
                          covariance_model : Covariance model ('exp' or 'gaus')
                          C0              : Variance parameter (fitted if None)
                          D               : Correlation length parameter (fitted if None)
                          fall_back_on_error : If True, fall back to nearest neighbor interpolation on error
                
        Returns
        -------
        Lon             : Longitude grid
        Lat             : Latitude grid
        data_interp     : Interpolated values (LSC with nearest neighbor extrapolation)
        raw_lsc         : Raw LSC interpolated values (may contain NaNs)
        C0             : Fitted or specified variance parameter
        D              : Fitted or specified correlation length parameter
        '''
        covariance_model = kwargs.get('covariance_model', 'exp')
        fall_back_on_error = kwargs.get('fall_back_on_error', False)
        
        # Validate covariance model
        valid_models = ['exp', 'gaus']
        if covariance_model not in valid_models:
            raise ValueError(f'covariance_model must be one of {valid_models}, got \'{covariance_model}\'.')
        
        # Extract coordinates and values
        lon = self.df_clean['lon'].values
        lat = self.df_clean['lat'].values
        values = self.values
        
        # Warn if duplicate points are present
        if self.verbose:
            coords = np.column_stack((lon, lat))
            _, idx, counts = np.unique(coords, axis=0, return_index=True, return_counts=True)
            if np.any(counts > 1):
                print(f"Warning: {np.sum(counts > 1)} duplicate points detected in input data.")
        
        # Use provided noise variance or default
        if N is None:
            N = np.ones_like(values) * 1e-6
        else:
            N = np.asarray(N)
            if N.shape != values.shape:
                raise ValueError(f"Noise variance N must have shape {values.shape}, got {N.shape}")
        
        # Grid points
        Xi = self.Lon.ravel()
        Yi = self.Lat.ravel()
        
        # Fit or use provided covariance parameters
        C0 = kwargs.get('C0', None)
        D = kwargs.get('D', None)
        
        try:
            if C0 is None or D is None:
                if robust_covariance:
                    from geoidlab.utils.lsc_helper_funcs import compute_spatial_covariance_robust as compute_cov
                else:
                    from geoidlab.utils.lsc_helper_funcs import compute_spatial_covariance as compute_cov
                covariance, covdist = compute_cov(lon, lat, values, chunk_size=chunk_size, use_chunking=use_chunking)
                if covariance_model == 'exp':
                    C0, D = fit_exponential_covariance(lon, lat, values, covariance, covdist)
                elif covariance_model == 'gaus':
                    C0, D = fit_gaussian_covariance(lon, lat, values, covariance, covdist)
        except Exception as e:
            if fall_back_on_error:
                if self.verbose:
                    print(f'Warning: Failed to fit covariance parameters: {str(e)}. Falling back to default parameters.')
                C0 = np.var(values)
                D = np.mean([self.grid_extent[1] - self.grid_extent[0], self.grid_extent[3] - self.grid_extent[2]])
            else:
                raise ValueError(
                    f'Failed to fit covariance parameters: {str(e)}. '
                    'Provide C0 and D in kwargs or enable fall_back_on_error.'
                )
        
        # Perform LSC with chunking and parallel processing
        try:
            if covariance_model == 'exp':
                zz = lsc_exponential(Xi, Yi, lon, lat, C0, D, N, values, 
                                   n_jobs=n_jobs, chunk_size=chunk_size, cache_dir=cache_dir, use_chunking=use_chunking)
            elif covariance_model == 'gaus':
                zz = lsc_gaussian(Xi, Yi, lon, lat, C0, D, N, values,
                                n_jobs=n_jobs, chunk_size=chunk_size, cache_dir=cache_dir, use_chunking=use_chunking)
        except Exception as e:
            if fall_back_on_error:
                if self.verbose:
                    print(f'Warning: LSC computation failed: {str(e)}. Falling back to nearest neighbor interpolation.')
                zz = self.neighbor_interp(np.column_stack((Xi, Yi)))
            else:
                raise ValueError(f'LSC computation failed: {str(e)}.')
        
        # Reshape to grid
        zz = zz.reshape(self.Lon.shape)
        
        # Apply nearest neighbor extrapolation for any NaN values
        z_nearest = self.neighbor_interp(np.column_stack((Xi, Yi))).reshape(self.Lon.shape)
        data_interp = np.where(np.isnan(zz), z_nearest, zz)
        
        return self.Lon, self.Lat, data_interp, zz, C0, D
    
    
    # def krigingInterpolant_chunked(
    #     self, 
    #     chunk_size: int = 100, 
    #     fall_back_on_error: bool = False, 
    #     merge: bool = True,
    #     coordinates_type: str = 'euclidean',
    #     **kwargs
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     '''
    #     Interpolate scattered data using Ordinary Kriging in a chunked fashion to optimize memory.
        
    #     Parameters
    #     ----------
    #     chunk_size : int
    #                  Number of grid rows per chunk.
    #     fall_back_on_error : bool
    #                  If True, falls back to default kriging parameters on error.
    #     merge      : bool
    #                  Whether to merge with nearest neighbor extrapolation for NaNs.
    #     kwargs     : additional parameters for OrdinaryKriging
        
    #     Returns
    #     -------
    #     Lon         : 2D array of longitude coordinates
    #     Lat         : 2D array of latitude coordinates
    #     data_interp : 2D array of interpolated values (after merging)
    #     zz          : 2D array of raw kriging values (possibly containing NaNs)
    #     ss          : 2D array of kriging variance
    #     '''
    #     from pykrige.ok import OrdinaryKriging
    #     # Default kriging parameters
    #     default_params = {
    #         'variogram_model': 'spherical',
    #         'nlags': 6,
    #         'verbose': False,
    #         'enable_plotting': False
    #     }
    #     kriging_params = default_params.copy()
    #     kriging_params.update(kwargs)
        
    #     lon = self.df_clean['lon'].values
    #     lat = self.df_clean['lat'].values
    #     # Convert lon to range [0, 360)
    #     lon = np.where(lon < 0, lon + 360, lon)
    #     # Print information to indicate creation of the kriging model
    #     if self.verbose:
    #         print('Creating Ordinary Kriging model...')
    #     try:
    #         ok = OrdinaryKriging(
    #             x=lon,
    #             y=lat,
    #             z=self.values,
    #             coordinates_type=coordinates_type,
    #             **kriging_params
    #         )
    #     except ValueError as e:
    #         if fall_back_on_error:
    #             if self.verbose:
    #                 print(f"Warning: {str(e)}. Falling back to default kriging parameters.")
    #             kriging_params = default_params.copy()
    #             ok = OrdinaryKriging(
    #                 x=lon,
    #                 y=lat,
    #                 z=self.values,
    #                 coordinates_type='geographic',
    #                 **kriging_params
    #             )
    #         else:
    #             raise ValueError(str(e))
        
    #     ny, nx = self.Lon.shape
    #     zz_full = np.full((ny, nx), np.nan)
    #     ss_full = np.full((ny, nx), np.nan)
        
    #     # Process grid rows in chunks for memory efficiency
    #     for i in range(0, ny, chunk_size):
    #         i_end = min(ny, i + chunk_size)
    #         lat_chunk = self.lat_grid[i:i_end]
    #         # Execute kriging on chunk; note: pykrige expects grid in order (x, y)
    #         zz_chunk, ss_chunk = ok.execute('grid', self.lon_grid, lat_chunk)
    #         zz_full[i:i_end, :] = zz_chunk
    #         ss_full[i:i_end, :] = ss_chunk
        
    #     if merge:
    #         z_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel())))
    #         z_nearest = z_nearest.reshape(self.Lon.shape)
    #         data_interp = np.where(np.isnan(zz_full), z_nearest, zz_full)
    #     else:
    #         data_interp = zz_full
        
    #     return self.Lon, self.Lat, data_interp, zz_full, ss_full
    
    def krigingInterpolant_chunked(
        self, 
        chunk_size: int = 100, 
        fall_back_on_error: bool = False, 
        merge: bool = True,
        coordinates_type: str = 'euclidean',
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Interpolate scattered data using Ordinary Kriging in a chunked fashion to optimize memory.
        
        Parameters
        ----------
        chunk_size : int
                    Number of grid rows per chunk.
        fall_back_on_error : bool
                    If True, falls back to default kriging parameters on error.
        merge      : bool
                    Whether to merge with nearest neighbor extrapolation for NaNs.
        kwargs     : additional parameters for OrdinaryKriging
        
        Returns
        -------
        Lon         : 2D array of longitude coordinates
        Lat         : 2D array of latitude coordinates
        data_interp : 2D array of interpolated values (after merging)
        zz          : 2D array of raw kriging values (possibly containing NaNs)
        ss          : 2D array of kriging variance
        '''
        from pykrige.ok import OrdinaryKriging
        # Default kriging parameters
        default_params = {
            'variogram_model': 'linear',
            'nlags': 6,
            'verbose': False,
            'enable_plotting': False
        }
        kriging_params = default_params.copy()
        kriging_params.update(kwargs)
        
        lon = self.df_clean['lon'].values
        lat = self.df_clean['lat'].values
        
        # Get variogram model from kwargs
        variogram_model = kwargs.get('variogram_model', 'linear')
        
        print(f'Fitting the {variogram_model} variogram model to the data...') if self.verbose else None
        
        if coordinates_type == 'geographic':
            print('Converting longitude to range [0, 360)')
            lon = np.where(lon < 0, lon + 360, lon)
        try:
            ok = OrdinaryKriging(
                x=lon,
                y=lat,
                z=self.values,
                coordinates_type=coordinates_type,
                **kriging_params
            )
        except ValueError as e:
            if fall_back_on_error:
                if self.verbose:
                    print(f"Warning: {str(e)}. Falling back to default kriging parameters.")
                kriging_params = default_params.copy()
                ok = OrdinaryKriging(
                    x=lon,
                    y=lat,
                    z=self.values,
                    coordinates_type=coordinates_type,
                    **kriging_params
                )
            else:
                raise ValueError(str(e))
        
        ny, nx = self.Lon.shape
        if self.verbose:
            print("Starting chunked kriging interpolation...")
            print(f"Grid size: {ny} rows x {nx} columns.")
            print(f"Processing {ny} rows in chunks of {chunk_size} rows each.")
        
        zz_full = np.full((ny, nx), np.nan)
        ss_full = np.full((ny, nx), np.nan)
        
        import time
        start_time = time.time()
        # Process grid rows in chunks for memory efficiency
        for i in range(0, ny, chunk_size):
            i_end = min(ny, i + chunk_size)
            lat_chunk = self.lat_grid[i:i_end]
            if self.verbose:
                print(f"Processing rows {i} to {i_end-1}...")
            # Execute kriging on chunk; note: pykrige expects grid in order (x, y)
            zz_chunk, ss_chunk = ok.execute('grid', self.lon_grid, lat_chunk)
            zz_full[i:i_end, :] = zz_chunk
            ss_full[i:i_end, :] = ss_chunk
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Chunked kriging interpolation completed in {total_time:.2f} seconds.")
        
        if merge:
            if self.verbose:
                print("Merging kriging results with nearest neighbor extrapolation for NaNs...")
            z_nearest = self.neighbor_interp(np.column_stack((self.Lon.ravel(), self.Lat.ravel())))
            z_nearest = z_nearest.reshape(self.Lon.shape)
            data_interp = np.where(np.isnan(zz_full), z_nearest, zz_full)
        else:
            data_interp = zz_full
        
        if self.verbose:
            print("Kriging chunked interpolation finished successfully.")
        
        return self.Lon, self.Lat, data_interp, zz_full, ss_full