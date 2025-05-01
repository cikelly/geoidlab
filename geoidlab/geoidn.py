############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import xarray as xr
# import warnings
# import bottleneck as bn

from geoidlab.utils.distances import haversine_fast, haversine_vectorized
from geoidlab.gravity import normal_gravity_somigliana
from geoidlab.constants import earth
from geoidlab.stokes_func import Stokes4ResidualGeoid

from tqdm import tqdm

class ResidualGeoid:
    '''
    Initialize the ResidualGeoid class.
    '''
    VALID_METHODS = {'hg', 'wg', 'og', 'ml'}  # Valid integration methods
    VALID_WINDOW_MODES = {'fixed', 'cap'}
    VALID_ELLIPSOIDS = {'wgs84', 'grs80'}
    DEFAULT_WINDOW_MODE = 'fixed'
    
    def __init__(
        self,
        res_anomaly: xr.Dataset,
        sph_cap: float = 1.0,
        sub_grid: tuple[float, float, float, float] = None,
        method: str = 'hg',
        ellipsoid: str = 'wgs84',
        nmax: int = None,
        window_mode: str = 'fixed'
    ) -> None:
        '''
        Initialize the ResidualGeoid class.
        
        Parameters
        ----------
        res_anomaly: gridded residual gravity anomalies
        sph_cap    : spherical cap for integration (degrees)
        sub_grid   : sub-grid to use for integration (min_lon, max_lon, min_lat, max_lat)
        method     : method for integration. Options are:
                        'hg' : Heck and Gruninger's modification
                        'wg' : Wong and Gore's modification
                        'og' : original Stokes' function
                        'ml' : Meissl's modification
        ellipsoid  : reference ellipsoid for normal gravity calculation
        nmax       : maximum degree of spherical harmonic expansion
        window_mode: window mode for integration. Options are: 'fixed' or 'cap'
        
        Returns
        -------
        None
        
        Reference
        ---------
        1. Hofmann-Wellenhof & Moritz (2005): Physical Geodesy (Section 2.21)
        '''
        # Validate sub-grid
        if sub_grid is None:
            raise ValueError('sub_grid must be provided')
        if len(sub_grid) != 4:
            raise ValueError('sub_grid must be a tuple of 4 values: (min_lon, max_lon, min')
        if not all(isinstance(x, (int, float)) for x in sub_grid):
            raise ValueError('All sub_grid values must be numeric')
        if sub_grid[0] >= sub_grid[1] or sub_grid[2] >= sub_grid[3]:
            raise ValueError('sub_grid must be a tuple of 4 values: (min_lon, max_lon, min_lat, max_lat)')
        
        # Validate method
        if method.lower() not in self.VALID_METHODS:
            raise ValueError(f'Invalid method: {method}. Must be one of {sorted(self.VALID_METHODS)}')
        
        # Validate nmax 
        if method.lower() in {'hg', 'wg'}:
            if nmax is None:
                raise ValueError(f'nmax must be provided for {method} method')
            if not isinstance(nmax, int) or nmax <=0:
                raise ValueError(f'nmax must be a positive integer, got {nmax}')
        
        # Validate ellipsoid
        if ellipsoid.lower() not in self.VALID_ELLIPSOIDS:
            raise ValueError(f'Invalid ellipsoid: {ellipsoid}. Must be one of {sorted(self.VALID_ELLIPSOIDS)}')
            
        # Validate res_anomaly
        if not isinstance(res_anomaly, xr.Dataset):
            raise TypeError('res_anomaly must be an xarray Dataset')
        if 'Dg' not in res_anomaly.data_vars:
            raise ValueError('res_anomaly must contain a \'Dg\' variable for gravity anomalies')
        if 'lon' not in res_anomaly.coords or 'lat' not in res_anomaly.coords:
            raise ValueError('res_anomaly must have \'lon\' and \'lat\' coordinates')
        
        # Validate window_mode
        if window_mode.lower() not in self.VALID_WINDOW_MODES:
            print(f'Warning: Invalid window_mode: \'{window_mode}\'. Setting window_mode=\'{self.DEFAULT_WINDOW_MODE}\'')
            window_mode = self.DEFAULT_WINDOW_MODE
            
        # Ensure sub_grid is within the bounds of res_anomaly
        min_lon, max_lon, min_lat, max_lat = sub_grid
        lon_min, lon_max = res_anomaly['lon'].min().item(), res_anomaly['lon'].max().item()
        lat_min, lat_max = res_anomaly['lat'].min().item(), res_anomaly['lat'].max().item()
        
        if (min_lon < lon_min or max_lon > lon_max or min_lat < lat_min or max_lat > lat_max):
            raise ValueError(f'sub_grid {sub_grid} is outside res_anomaly bounds (lon: [{lon_min}, {lon_max}], lat: [{lat_min}, {lat_max}])')
        
        # Validate spherical cap (sph_cap)
        if not isinstance(sph_cap, (int, float)) or sph_cap <= 0:
            raise ValueError(f'sph_cap must be a positive number, got {sph_cap}')
        
        # Store validated parameters
        self.sub_grid = sub_grid
        self.res_anomaly = res_anomaly
        self.sph_cap = sph_cap
        self.method = method
        self.ellipsoid = ellipsoid
        self.nmax = nmax
        self.window_mode = window_mode
        
        # Grid information
        self.Lon, self.Lat = np.meshgrid(res_anomaly['lon'].values, res_anomaly['lat'].values)
        self.nrows, self.ncols = res_anomaly['Dg'].shape
        self.dphi = (lat_max - lat_min) / (self.nrows - 1)
        self.dlam = (lon_max - lon_min) / (self.ncols - 1)
        
        # Extract sub-grid residual anomalies
        self.res_anomaly_P = self.res_anomaly.sel(
            lon=slice(min_lon, max_lon), 
            lat=slice(min_lat, max_lat)
        )
        
        self.LonP, self.LatP = np.meshgrid(self.res_anomaly_P['lon'].values, self.res_anomaly_P['lat'].values)
        self.nrows_P, self.ncols_P = self.res_anomaly_P['Dg'].shape
        self.phip = np.radians(self.LatP)
        self.lonp = np.radians(self.LonP)
        
        # Normal gravity (Somigliana's method)
        self.gamma_0  = normal_gravity_somigliana(phi=self.LatP, ellipsoid=self.ellipsoid)
        self.gamma_0 *= 1e5  # Convert to mGal
        
        # Precompute constants
        self.R = earth()['radius'] # Earth radius
        
    def stokes_kernel(self) -> np.ndarray:
        '''
        Compute Stokes' kernel based on the selected method
        
        Returns
        -------
        S_k     : Stokes' kernel values
        '''
        method_map = {
            'og': lambda: self.stokes_calculator.stokes()[0],
            'wg': lambda: self.stokes_calculator.wong_and_gore(),
            'hg': lambda: self.stokes_calculator.heck_and_gruninger(),
            'ml': lambda: self.stokes_calculator.meissl()
        }
        
        S_k = method_map[self.method]()
        
        # Set NaN and +/- inf values to 0
        S_k = np.nan_to_num(S_k, nan=0.0, posinf=0.0, neginf=0.0)
        
        return S_k

    def compute_geoid(self) -> np.ndarray:
        '''
        Compute the residual geoid height
        
        Returns
        -------
        N_res  : Residual geoid height in meters
        '''
        cosphip = np.cos(self.phip)
        
        # Near zone computation
        self.N_inner = self.R / self.gamma_0 * np.sqrt(
            cosphip * np.radians(self.dphi) * np.radians(self.dlam) / np.pi
        ) * self.res_anomaly_P['Dg'].values
        
        # Far zone computation
        self.N_far = np.zeros_like(self.N_inner)
        psi0 = np.radians(self.sph_cap)
        
        # Precompute np.radians(self.dphi) / 2 and np.radians(self.dlam) / 2
        dphi_2 = np.radians(self.dphi) / 2
        dlam_2 = np.radians(self.dlam) / 2
        
        if self.window_mode == 'fixed':
            # Sub-window for integration
            dn = round(self.nrows - self.nrows_P) + 1
            dm = round(self.ncols - self.ncols_P) + 1
            n1 = 0
            n2 = dn
            
            for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
                m1 = 0
                m2 = dm
                for j in range(self.ncols_P):
                    # Extract window data
                    smallDg = self.res_anomaly['Dg'].values[n1:n2, m1:m2]
                    smallphi = np.radians(self.Lat[n1:n2, m1:m2])
                    smalllon = np.radians(self.Lon[n1:n2, m1:m2])
                    
                    # Compute surface area on the sphere
                    lat1 = smallphi - dphi_2
                    lat2 = smallphi + dphi_2
                    lon1 = smalllon - dlam_2
                    lon2 = smalllon + dlam_2
                    A_k = self.R**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))
                    
                    self.stokes_calculator = Stokes4ResidualGeoid(
                        lonp=self.lonp[i, j],
                        latp=self.phip[i, j],
                        lon=smalllon,
                        lat=smallphi,
                        psi0=psi0,
                        nmax=self.nmax
                    )
                    
                    # S_k = self.stokes_calculator.stokes()[0]
                    S_k = self.stokes_kernel()
                    
                    # Spherical Distance
                    sd = haversine_fast(
                        lon1=np.degrees(self.lonp[i, j]),
                        lat1=np.degrees(self.phip[i, j]),
                        lon2=np.degrees(smalllon),
                        lat2=np.degrees(smallphi),
                        unit='deg'
                    )
                    
                    # Prone to rounding errors
                    # sd = haversine_fast(
                    #     lon1=self.lonp[i, j] * 180. / np.pi,
                    #     lat1=self.phip[i, j] * 180. / np.pi,
                    #     lon2=smalllon * 180. / np.pi,
                    #     lat2=smallphi * 180. / np.pi,
                    #     unit='deg'
                    # )
                    
                    sd[sd > self.sph_cap] = np.nan
                    
                    # Mask S_k for points beyond self.scap
                    S_k[np.isnan(sd)] = np.nan
                    
                    # Outer (far) zone
                    c_k = A_k * S_k
                    self.N_far[i, j] = np.nansum(c_k * smallDg) * 1 / (4 * np.pi * self.gamma_0[i, j] * self.R)
                    
                    # Move window
                    m1 += 1
                    m2 += 1
                # Move window
                n1 += 1
                n2 += 1
        else:
            # Find sub-grid offset in full grid
            lat = self.res_anomaly['lat'].values
            lon = self.res_anomaly['lon'].values
            lat_sub = self.res_anomaly_P['lat'].values
            lon_sub = self.res_anomaly_P['lon'].values
            row_start = np.where(lat == lat_sub[0])[0][0]
            col_start = np.where(lon == lon_sub[0])[0][0]
            
            for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
                for j in range(self.ncols_P):
                    # Map sub-grid to full grid indices
                    row_p = row_start + i
                    col_p = col_start + j
                    
                    # Get cap window bounds and mask
                    mask, (row_min, row_max, col_min, col_max) = self.get_cap_window(row_p, col_p)
                    
                    # Extract window data
                    smallDg = self.res_anomaly['Dg'].values[row_min:row_max, col_min:col_max]
                    smallphi = np.radians(self.Lat[row_min:row_max, col_min:col_max])
                    smalllon = np.radians(self.Lon[row_min:row_max, col_min:col_max])
                    
                    # Apply mask to windowed data
                    smallDg = np.where(mask, smallDg, np.nan)
                    smallphi = np.where(mask, smallphi, np.nan)
                    smalllon = np.where(mask, smalllon, np.nan)
                    
                    # Compute surface area on the sphere
                    lat1 = smallphi - dphi_2
                    lat2 = smallphi + dphi_2
                    lon1 = smalllon - dlam_2
                    lon2 = smalllon + dlam_2
                    A_k = self.R**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))
                    A_k = np.where(np.isfinite(smallDg), A_k, np.nan)


                    self.stokes_calculator = Stokes4ResidualGeoid(
                        lonp=self.lonp[i, j],
                        latp=self.phip[i, j],
                        lon=smalllon,
                        lat=smallphi,
                        psi0=psi0,
                        nmax=self.nmax
                    )

                    S_k = self.stokes_kernel()
                    # Apply mask to S_k (use mask from get_cap_window)
                    S_k = np.where(mask, S_k, np.nan)
                    
                    # Outer (far) zone
                    c_k = A_k * S_k
                    self.N_far[i, j] = np.nansum(c_k * smallDg) * 1 / (4 * np.pi * self.gamma_0[i, j] * self.R)

        N_res = self.N_inner + self.N_far
        self.N_res = N_res

        return N_res

    def get_cap_window(self, row_p, col_p) -> tuple[bool, tuple[int, int, int, int]]:
        '''
        Get indices of points within spherical cap

        Parameters
        ----------
        row_p       : row index of computation point
        col_p       : column index of computation point
        
        Returns
        -------
        mask        : boolean mask of points within spherical cap
        bounds      : (row_min, row_max, col_min, col_max) of window
        
        Notes
        -----
        This approach is a hybrid of a rectangular window and a circular window.
        We start with a rectangular window adjusted for the latitude of the computation point.
        Then we refined the selection by excluding points outside the spherical cap.
        '''
        # Latitude of computation point
        lat_p = self.Lat[row_p, col_p]
        lon_p = self.Lon[row_p, col_p]
        
        # Estimate window size based on latitude
        dn = int(np.ceil(self.sph_cap / self.dphi))
        dm = int(np.ceil(self.sph_cap / (self.dlam * np.cos(np.radians(lat_p)))))
        
        # Define window bounds
        row_min = max(0, row_p - dn)
        row_max = min(self.nrows, row_p + dn + 1)
        col_min = max(0, col_p - dm)
        col_max = min(self.ncols, col_p + dm + 1)
        
        # Extract window coordinates
        lat_window = self.Lat[row_min:row_max, col_min:col_max]
        lon_window = self.Lon[row_min:row_max, col_min:col_max] 
        
        # Compute distances within window
        distances = haversine_vectorized(lon_p, lat_p, lon_window, lat_window)
        
        # Mask points within spherical cap
        mask = distances <= self.sph_cap
        
        return mask, (row_min, row_max, col_min, col_max)