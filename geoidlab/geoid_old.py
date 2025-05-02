############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import xarray as xr
import warnings
import bottleneck as bn

from geoidlab.utils.distances import haversine
from geoidlab.gravity import normal_gravity_somigliana
from geoidlab.constants import earth
from geoidlab.stokes_func import Stokes4ResidualGeoid

from tqdm import tqdm



class ResidualGeoid:
    '''
    Geoid class for modeling the geoid via the remove-compute-restore method.
    '''
    VALID_METHODS = {'hg', 'wg', 'og', 'ml'}  # Valid integration methods
    VALID_WINDOW_MODES = {'small', 'cap'}
    VALID_ELLIPSOIDS = {'wgs84', 'grs80'}
    
    def __init__(
        self,
        res_anomaly: xr.Dataset,
        sph_cap: float = 1.0,
        sub_grid: tuple[float, float, float, float] = None,
        method: str = 'hg',
        ellipsoid: str = 'wgs84',
        nmax: int = None,
        window_mode: str = 'small',
        overwrite: bool = True
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
                    (required for 'hg' and 'wg' methods)
        window_mode: window mode for integration. Options are: 'small' or 'cap'
        overwrite  : whether to overwrite existing data files
        
        Returns
        -------
        None
        
        Reference
        ---------
        1. Hofmann-Wellenhof & Moritz (2005): Physical Geodesy (Section 2.21)
        '''
        # Validate window_mode
        window_mode = window_mode.lower()
        if window_mode not in self.VALID_WINDOW_MODES:
            raise ValueError(f'Invalid window_mode: {window_mode}. Must be one of {sorted(self.VALID_WINDOW_MODES)}')
        
        # Validate sub_grid
        if sub_grid is None:
            raise ValueError('sub_grid must be provided')
        if not isinstance(sub_grid, (tuple, list)) or len(sub_grid) != 4:
            raise ValueError('sub_grid must be a tuple/list of 4 values: (min_lon, max_lon, min_lat, max_lat)')
        min_lon, max_lon, min_lat, max_lat = sub_grid
        if not all(isinstance(x, (int, float)) for x in sub_grid):
            raise ValueError('All sub_grid values must be numeric')
        if min_lon >= max_lon:
            raise ValueError(f'min_lon ({min_lon}) must be less than max_lon ({max_lon})')
        if min_lat >= max_lat:
            raise ValueError(f'min_lat ({min_lat}) must be less than max_lat ({max_lat})')
            
        # Validate method
        method = method.lower()
        if method not in self.VALID_METHODS:
            raise ValueError(f'Invalid method: {method}. Must be one of {sorted(self.VALID_METHODS)}')
            
        # Validate nmax for methods that require it
        if method in {'hg', 'wg'}:
            if nmax is None:
                raise ValueError(f'nmax must be provided for method {method}')
            if not isinstance(nmax, int) or nmax <= 0:
                raise ValueError(f'nmax must be a positive integer, got {nmax}')
        
        # Validate ellipsoid
        if ellipsoid.lower() not in self.VALID_ELLIPSOIDS:
            raise ValueError(f'Invalid ellipsoid: {ellipsoid}. Supported ellipsoids: {sorted(self.VALID_ELLIPSOIDS)}')
        
        # Validate res_anomaly
        if not isinstance(res_anomaly, xr.Dataset):
            raise TypeError('res_anomaly must be an xarray Dataset')
        if 'Dg' not in res_anomaly.data_vars:
            raise ValueError('res_anomaly must contain a \'Dg\' variable for gravity anomalies')
        if 'lon' not in res_anomaly.coords or 'lat' not in res_anomaly.coords:
            raise ValueError('res_anomaly must have \'lon\' and \'lat\' coordinates')
        
        # Ensure sub_grid is within the bounds of res_anomaly
        lon_min, lon_max = res_anomaly['lon'].min().item(), res_anomaly['lon'].max().item()
        lat_min, lat_max = res_anomaly['lat'].min().item(), res_anomaly['lat'].max().item()
        if (min_lon < lon_min or max_lon > lon_max or min_lat < lat_min or max_lat > lat_max):
            raise ValueError(f'sub_grid {sub_grid} is outside res_anomaly bounds (lon: [{lon_min}, {lon_max}], lat: [{lat_min}, {lat_max}])')
        
        # Store validated parameters as float64/int
        self.sub_grid = tuple(float(x) for x in sub_grid)
        self.res_anomaly = res_anomaly
        self.sph_cap = np.float64(sph_cap)
        self.method = method
        self.ellipsoid = ellipsoid.lower()
        self.nmax = int(nmax) if nmax is not None else None
        self.window_mode = window_mode
        
        # Extract coordinates and convert to float64
        lon = res_anomaly.lon.values.astype(np.float64)
        lat = res_anomaly.lat.values.astype(np.float64)

        # Grid size
        self.nrows, self.ncols = self.res_anomaly['Dg'].shape
        self.dlam = np.float64((max(lon) - min(lon)) / (self.ncols - 1))
        self.dphi = np.float64((max(lat) - min(lat)) / (self.nrows - 1))
        
        # Extract sub-grid residual anomalies
        self.res_anomaly_P = self.res_anomaly.sel(
            lon=slice(self.sub_grid[0], self.sub_grid[1]), 
            lat=slice(self.sub_grid[2], self.sub_grid[3])
        )
        
        # Create meshgrids - convert inputs to float64 first
        lon_p = self.res_anomaly_P['lon'].values.astype(np.float64)
        lat_p = self.res_anomaly_P['lat'].values.astype(np.float64)
        self.LonP, self.LatP = np.meshgrid(lon_p, lat_p)
        
        # Calculate normal gravity at the ellipsoid
        self.gamma_0 = normal_gravity_somigliana(phi=self.LatP, ellipsoid=self.ellipsoid)
        self.gamma_0 = (self.gamma_0 * 1e5).astype(np.float64)  # Convert to mGal
        
        # Create full grid meshgrid
        lon_full = self.res_anomaly['lon'].values.astype(np.float64)
        lat_full = self.res_anomaly['lat'].values.astype(np.float64)
        self.Lon, self.Lat = np.meshgrid(lon_full, lat_full)
        
        # Pre-compute constants
        self.R = np.float64(earth()['radius'])  # Earth radius
        self.k = (1 / (4 * np.pi * self.gamma_0 * self.R)).astype(np.float64)
        
        # Window size calculation: Precompute window size
        self.nrows_P, self.ncols_P = self.res_anomaly_P['Dg'].shape
        if self.window_mode == 'small':
            self.dn = int(np.round(self.ncols - self.ncols_P)) + 1
            self.dm = int(np.round(self.nrows - self.nrows_P)) + 1
        else:
            # Window size based on spherical cap
            self.dn = int(np.ceil(self.sph_cap / self.dlam)) * 2 + 1 # Ensure cap is fully covered
            self.dm = int(np.ceil(self.sph_cap / self.dphi)) * 2 + 1
            # Ensure window does not exceed full grid
            self.dn = min(self.dn, self.ncols)
            self.dm = min(self.dm, self.nrows)
        
        # Pre-allocate arrays for compute_geoid
        self.N_inner = np.zeros((self.nrows_P, self.ncols_P), dtype=np.float64)
        self.N_far = np.zeros_like(self.N_inner)
        self._smallDg = np.empty((self.dm, self.dn), dtype=np.float64)  # Changed from (dn, dm)
        self._smallphi = np.empty_like(self._smallDg)
        self._smalllon = np.empty_like(self._smallDg)
        self._A_k = np.empty_like(self._smallDg)
        self._lat1 = np.empty_like(self._smallDg)
        self._lat2 = np.empty_like(self._smallDg)
        self._lon1 = np.empty_like(self._smallDg)
        self._lon2 = np.empty_like(self._smallDg)

    def stokes_kernel(self) -> np.ndarray:
        '''
        Compute Stokes' kernel based on the selected method
        
        Parameters
        ----------
        sin2_psi_2 : sin²(ψ/2) values
        cos_psi    : cos(ψ) values
        
        Returns
        -------
        S_k : Stokes' kernel values
        '''
        
        method_map = {
            'og': self.stokes_calculator.stokes(),
            'wg': self.stokes_calculator.wong_and_gore(),
            'hg': self.stokes_calculator.heck_and_gruninger(),
            'ml': self.stokes_calculator.meissl()
        }
        
        if self.method not in method_map:
            raise ValueError(f'Unknown method: {self.method}')
        
        # Handle any numerical issues
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            
            S_k = method_map[self.method]
            S_k = np.nan_to_num(S_k, nan=0.0)
            
        return S_k

    def get_window(
        self, 
        i: int, 
        j: int
    ) -> None:
        '''
        Compute windowed data based on window mode ('small' or 'cap')
        '''
        i_center = int(np.round((self.LatP[i, j] - self.res_anomaly['lat'].values[0]) / self.dphi))
        j_center = int(np.round((self.LonP[i, j] - self.res_anomaly['lon'].values[0]) / self.dlam))
        i_start = max(0, i_center - self.dm // 2)
        i_end = min(self.nrows, i_start + self.dm)
        j_start = max(0, j_center - self.dn // 2)
        j_end = min(self.ncols, j_start + self.dn)
        i_start = max(0, i_end - self.dm)
        j_start = max(0, j_end - self.dn)
        
        # Extract window data
        window_Dg = self.res_anomaly['Dg'].values[i_start:i_end, j_start:j_end]
        window_phi = np.radians(self.Lat[i_start:i_end, j_start:j_end])
        window_lon = np.radians(self.Lon[i_start:i_end, j_start:j_end])
        if window_Dg.shape != (self.dm, self.dn):
            di, dj = window_Dg.shape
            pad_width = ((0, self.dm - di), (0, self.dn - dj))
            window_Dg = np.pad(window_Dg, pad_width, mode='constant', constant_values=np.nan)
            window_phi = np.pad(window_phi, pad_width, mode='constant', constant_values=np.nan)
            window_lon = np.pad(window_lon, pad_width, mode='constant', constant_values=np.nan)
        np.copyto(self._smallDg, window_Dg)
        np.copyto(self._smallphi, window_phi)
        np.copyto(self._smalllon, window_lon)
    
    def compute_geoid(self) -> np.ndarray:
        '''
        Compute the residual geoid height
        '''
        phip = np.radians(self.LatP)
        lonp = np.radians(self.LonP)
        cosphip = np.cos(phip)
        
        # Near zone computation
        self.N_inner = self.R / self.gamma_0 * np.sqrt(
            cosphip * np.radians(self.dphi) * np.radians(self.dlam) / np.pi
        ) * self.res_anomaly_P['Dg'].values
        
        # self.N_inner = (self.R / (2 * self.gamma_0)) * np.sqrt(
        #     cosphip * np.radians(self.dphi) * np.radians(self.dlam)
        # ) * self.res_anomaly_P['Dg'].values
        
        # Far zone computation
        psi0 = np.radians(self.sph_cap)
        dm, dn = self._smallDg.shape
        
        # if self.window_mode == 'small':
        # n1 = 0
        # n2 = self.dm
        for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
            for j in range(self.ncols_P):
                # Extract window data
                self.get_window(i, j)
                # Surface area
                np.subtract(self._smallphi, np.radians(self.dphi) / 2, out=self._lat1)
                np.add(self._smallphi, np.radians(self.dphi) / 2, out=self._lat2)
                np.subtract(self._smalllon, np.radians(self.dlam) / 2, out=self._lon1)
                np.add(self._smalllon, np.radians(self.dlam) / 2, out=self._lon2)
                np.multiply(
                    self.R**2,
                    np.abs(self._lon2 - self._lon1) * np.abs(np.sin(self._lat2) - np.sin(self._lat1)),
                    out=self._A_k
                )
                
                # Stokes' kernel
                self.stokes_calculator = Stokes4ResidualGeoid(
                    lonp=lonp[i,j],
                    latp=phip[i,j],
                    lon=self._smalllon.flatten(),
                    lat=self._smallphi.flatten(),
                    psi0=psi0,
                    nmax=self.nmax
                )
                S_k = self.stokes_kernel() if self.method != 'og' else self.stokes_kernel()[0]
                S_k = S_k.reshape(self._smallDg.shape)
                
                # Spherical distance
                sd = haversine(
                    np.degrees(lonp[i, j]),
                    np.degrees(phip[i, j]),
                    np.degrees(self._smalllon),
                    np.degrees(self._smallphi),
                    unit='deg'
                )
                sd[sd > self.sph_cap] = np.nan
                sd[sd == 0] = np.nan
                S_k[np.isnan(sd)] = np.nan
                
                # Far zone contribution
                self.N_far[i, j] = bn.nansum(self._A_k * S_k * self._smallDg) * self.k[i, j]
        # else:
        #     # Window size based on spherical cap
        #     for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
        #         for j in range(self.ncols_P):
        #             self.get_window(i, j)
        #             # Surface area
        #             np.subtract(self._smallphi, np.radians(self.dphi) / 2, out=self._lat1)
        #             np.add(self._smallphi, np.radians(self.dphi) / 2, out=self._lat2)
        #             np.subtract(self._smalllon, np.radians(self.dlam) / 2, out=self._lon1)
        #             np.add(self._smalllon, np.radians(self.dlam) / 2, out=self._lon2)
        #             np.multiply(
        #                 self.R**2,
        #                 np.abs(self._lon2 - self._lon1) * np.abs(np.sin(self._lat2) - np.sin(self._lat1)),
        #                 out=self._A_k
        #             )

        #             # Stokes' kernel
        #             self.stokes_calculator = Stokes4ResidualGeoid(
        #                 lonp=lonp[i,j],
        #                 latp=phip[i,j],
        #                 lon=self._smalllon.flatten(),
        #                 lat=self._smallphi.flatten(),
        #                 psi0=psi0,
        #                 nmax=self.nmax
        #             )
        #             S_k = self.stokes_kernel() if self.method != 'og' else self.stokes_kernel()[0]
        #             S_k = S_k.reshape(self._smallDg.shape)

        #             # Spherical distance
        #             sd = haversine(
        #                 np.degrees(lonp[i, j]),
        #                 np.degrees(phip[i, j]),
        #                 np.degrees(self._smalllon),
        #                 np.degrees(self._smallphi),
        #                 unit='deg'
        #             )
        #             sd[sd > self.sph_cap] = np.nan
        #             sd[sd == 0] = np.nan
        #             S_k[np.isnan(sd)] = np.nan

        #             # Far zone contribution
        #             self.N_far[i, j] = bn.nansum(self._A_k * S_k * self._smallDg) * self.k[i, j]
        
        # Residual geoid: Combine near and far zone contributions
        N_res = self.N_inner + self.N_far
        
        return N_res