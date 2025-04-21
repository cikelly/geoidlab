############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import xarray as xr
import warnings
import bottleneck as bn

from .utils.distances import haversine
from .gravity import normal_gravity_somigliana
from .constants import earth
from .stokes_func import Stokes4ResidualGeoid
from tqdm import tqdm



class ResidualGeoid:
    '''
    Geoid class for modeling the geoid via the remove-compute-restore method.
    '''
    VALID_METHODS = {'hg', 'wg', 'og', 'ml'}  # Valid integration methods
    
    def __init__(
        self,
        res_anomaly: xr.Dataset,
        sph_cap: float = 1.0,
        sub_grid: tuple[float, float, float, float] = None,
        method: str = 'hg',
        ellipsoid: str = 'wgs84',
        nmax: int = None,
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
        '''
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
            
        # Store validated parameters as float64/int
        self.sub_grid = tuple(float(x) for x in sub_grid)
        self.res_anomaly = res_anomaly
        self.sph_cap = np.float64(sph_cap)
        self.method = method
        self.ellipsoid = ellipsoid.lower()
        self.nmax = int(nmax) if nmax is not None else None
        
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
        self.gamma0 = normal_gravity_somigliana(phi=self.LatP, ellipsoid=self.ellipsoid)
        self.gamma0 = (self.gamma0 * 1e5).astype(np.float64)  # Convert to mGal
        
        # Create full grid meshgrid
        lon_full = self.res_anomaly['lon'].values.astype(np.float64)
        lat_full = self.res_anomaly['lat'].values.astype(np.float64)
        self.Lon, self.Lat = np.meshgrid(lon_full, lat_full)
        
        # Pre-compute constants
        self.R = np.float64(earth()['radius'])  # Earth radius
        self.k = (1 / (4 * np.pi * self.gamma0 * self.R)).astype(np.float64)
        
        # Pre-allocate arrays for compute_geoid
        self.nrows_P, self.ncols_P = self.res_anomaly_P['Dg'].shape
        self.N_inner = np.zeros((self.nrows_P, self.ncols_P), dtype=np.float64)
        self.N_far = np.zeros_like(self.N_inner)
        
        # Pre-allocate work arrays for compute_geoid
        dn = int(np.round(self.ncols - self.ncols_P)) + 1
        dm = int(np.round(self.nrows - self.nrows_P)) + 1
        self._smallDg = np.empty((dn, dm), dtype=np.float64)
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

    def compute_geoid(self) -> np.ndarray:
        '''
        Compute the residual geoid height
        '''
        phip = np.radians(self.LatP)
        lonp = np.radians(self.LonP)
        cosphip = np.cos(phip)
        
        # Near zone computation
        self.N_inner = self.R / self.gamma0 * np.sqrt(
            cosphip * np.radians(self.dphi) * np.radians(self.dlam) / np.pi
        ) * self.res_anomaly_P['Dg'].values
        
        # Far zone computation using preallocated arrays
        n1 = 0
        n2 = self._smallDg.shape[0]
        psi0 = np.radians(self.sph_cap)
        
        for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
            m1 = 0
            m2 = self._smallDg.shape[1]
            for j in range(self.ncols_P):
                # Use np.copyto to populate preallocated arrays
                np.copyto(self._smallDg, self.res_anomaly['Dg'].values[n1:n2, m1:m2])
                np.copyto(self._smallphi, np.radians(self.Lat[n1:n2, m1:m2]))
                np.copyto(self._smalllon, np.radians(self.Lon[n1:n2, m1:m2]))
                
                # Surface area on the sphere using preallocated arrays
                np.subtract(self._smallphi, np.radians(self.dphi) / 2, out=self._lat1)
                np.add(self._smallphi, np.radians(self.dphi) / 2, out=self._lat2)
                np.subtract(self._smalllon, np.radians(self.dlam) / 2, out=self._lon1)
                np.add(self._smalllon, np.radians(self.dlam) / 2, out=self._lon2)
                
                np.multiply(
                    self.R**2,
                    np.abs(self._lon2 - self._lon1) * np.abs(np.sin(self._lat2) - np.sin(self._lat1)),
                    out=self._A_k
                )

                # Compute Stokes' kernel
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
                # Mask points outside the spherical cap
                sd[sd > self.sph_cap] = np.nan
                S_k[np.isnan(sd)] = np.nan
                
                # Compute contribution
                self.N_far[i, j] = bn.nansum(self._A_k * S_k * self._smallDg) * self.k[i, j]
                
                m1 += 1
                m2 += 1
            n1 += 1
            n2 += 1
        
        return self.N_inner + self.N_far