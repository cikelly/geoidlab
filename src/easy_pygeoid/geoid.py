############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import xarray as xr

from .utils.distances import haversine
from .gravity import normal_gravity_somigliana
from .constants import earth


class ResidualGeoid:
    '''
    Geoid class for modeling the geoid via the remove-compute-restore
    method.
    '''
    def __init__(
        self,
        res_anomaly: xr.Dataset,
        sph_cap: float = 1.0,
        sub_grid: tuple[float, float, float, float] = None,
        bbox_off: float = 1.,
        method:str = 'hk',
        ellipsoid: str = 'wgs84'
    ) -> None:
        '''
        Initialize the ResidualGeoid class.
        
        Parameters
        ----------
        res_anomaly: gridded residual gravity anomalies
        sph_cap    : spherical cap for integration (degrees)
        sub_grid   : sub-grid to use for integration
        bbox_off   : offset for bounding box (in km)
        method     : method for integration. Options are
                        'hk' for Heck and Gruninger's modification
                        'wg' for Wong and Gore's modification
                        'og' for original Stokes' function
        ellipsoid  : reference ellipsoid for normal gravity calculation
                    
        Returns
        -------
        None
        
        Notes
        -----
        1. Ensure that varnames in res_anomaly are 'Dg', 'lat', and 'lon'
        2. The sub-grid is defined as (min_lon, max_lon, min_lat, max_lat)
        3. The bounding box offset is in kilometers
        4. The method is case-insensitive
        5. The ellipsoid is case-insensitive
        6. The residual anomalies should be in mGal
        '''
        self.res_anomaly = res_anomaly
        self.sph_cap = sph_cap
        self.method = method.lower()
        self.ellipsoid = ellipsoid
        lon = self.res_anomaly.lon.values
        lat = self.res_anomaly.lat.values
        
        
        if sub_grid is None:
            from .terrain import TerrainQuantities
            
            box_off_deg = TerrainQuantities.km2deg(bbox_off)
            
            min_lon = lon.min().values - self.sph_cap,
            max_lon = lon.max().values + self.sph_cap,
            min_lat = lat.min().values - self.sph_cap,
            max_lat = lat.max().values + self.sph_cap
            
            # Ensure sub-grid is within bounds
            if min_lat >= min(lat) or max_lat >= max(lat) or min_lon >= min(lon) or max_lon <= max(lon):
                self.sub_grid = (
                    min(lon)+box_off_deg, 
                    max(lon)-box_off_deg, 
                    min(lat)+box_off_deg, 
                    max(lat)-box_off_deg
                )
            else:
                self.sub_grid = (min_lon, max_lon, min_lat, max_lat)
        else:
            self.sub_grid = sub_grid
        
        # Extract sub-grid residual anomalies
        self.res_anomaly_P = self.res_anomaly.sel(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3])

        # Grid size
        self.nrows, self.ncols = self.res_anomaly['Dg'].shape
        self.dlam = (max(lon) - min(lon)) / (self.ncols - 1)
        self.dphi = (max(lat) - min(lat)) / (self.nrows - 1)
        
        # Calculate normal gravity at the ellipsoid
        self.LonP, LatP = np.meshgrid(self.res_anomaly_P['lon'], self.res_anomaly_P['lat'])  
        self.gamma0 = normal_gravity_somigliana(phi=LatP, ellipsoid=self.ellipsoid)
        self.gamma0 *= 1e5
        self.LatP = LatP
        
    
    def compute_geoid(self) -> np.ndarray:
        '''
        Compute the residual geoid height
        '''
        phip = np.radians(self.LatP)
        lambdap = np.radians(self.LonP)
        
        # Near zone computation
        psi_0 = np.sqrt( (np.cos(phip) * np.radians(self.dphi)  * np.radians(self.dlam)) * 1 / np.pi )
        s_0 = earth['radius'] * psi_0
        N_inner = s_0 / self.gamma0 * self.res_anomaly_P['Dg']
        
        # Far zone computation
        N_far = np.zeros_like(N_inner)

        # Precompute coordinates for efficiency
        lon2 = np.radians(self.res_anomaly['lon'].values)  # Full grid longitudes
        lat2 = np.radians(self.res_anomaly['lat'].values)  # Full grid latitudes
        Lon2, Lat2 = np.meshgrid(lon2, lat2)  # Full grid mesh
        
        # Area element per cell (in m**2)
        lat1 = Lat2 - np.radians(self.dphi) / 2
        lat2 = Lat2 + np.radians(self.dphi) / 2
        lon1 = Lon2 - np.radians(self.dlam) / 2
        lon2 = Lon2 + np.radians(self.dlam) / 2
        dA = earth['radius']**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))
        
        for i in range(self.nrows):
            for j in range(self.ncols):
                # Computation point
                lon1 = lambdap[i, j]
                lat1 = phip[i, j]
                
                # Spherical distance to all points in full grid
                sd = haversine(lon1, lat1, Lon2, Lat2, r=1.0, unit='rad')

                cos_psi = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
                psi = np.arccos(np.clip(cos_psi, -1, 1))  # Ensure numerical stability

                # Apply spherical cap
                mask = sd <= np.radians(self.sph_cap)
                psi = sd[mask]
                if not np.any(mask):  # Skip if no points within cap
                    continue
                
                # Stokes function
                S_k = self._stokes_function(psi)
                
                # Far zone contribution
                dg_subset = self.res_anomaly['Dg'].values[mask]
                dA_subset = dA[mask]
                N_far[i, j] = np.nansum(S_k * dg_subset * dA_subset) / (4 * np.pi * self.gamma0[i, j] * earth['radius'])
        
        # Total geoid undulation
        N = N_inner + N_far

        return N
    
    def _stokes_function(self, psi: np.ndarray) -> np.ndarray:
        '''
        Compute Stokes' function for the given spherical distance psi.

        Parameters
        ----------
        psi : spherical distance in radians

        Returns
        -------
        S_k : Stokes' function values
        '''
        if self.method == 'hk':
            # Heck and Gruninger's modification
            S_k = 1 / np.sin(psi / 2) - 6 * np.sin(psi / 2) + 1 - 5 * np.cos(psi) - 3 * np.cos(psi) * np.log(np.sin(psi / 2) + np.sin(psi / 2)**2)
        elif self.method == 'wg':
            # Wong and Gore's modification
            S_k = 1 / np.sin(psi / 2) - 6 * np.sin(psi / 2) + 1 - 5 * np.cos(psi)
        elif self.method == 'og':
            # Original Stokes' function
            S_k = 1 / np.sin(psi / 2) - 6 * np.sin(psi / 2) + 1 - 5 * np.cos(psi) - 3 * np.cos(psi) * np.log(np.sin(psi / 2))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return S_k