############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import xarray as xr
import warnings

from .utils.distances import haversine
from .gravity import normal_gravity_somigliana
from .constants import earth
from tqdm import tqdm



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
        
        if sub_grid is None:
            raise ValueError('sub_grid must be provided')
        
        self.sub_grid = sub_grid
        self.res_anomaly = res_anomaly
        self.sph_cap     = sph_cap
        self.method      = method.lower()
        self.ellipsoid   = ellipsoid
        lon              = self.res_anomaly.lon.values
        lat              = self.res_anomaly.lat.values

        # Grid size
        self.nrows, self.ncols = self.res_anomaly['Dg'].shape
        self.dlam = (max(lon) - min(lon)) / (self.ncols - 1)
        self.dphi = (max(lat) - min(lat)) / (self.nrows - 1)
        
        # Extract sub-grid residual anomalies
        self.res_anomaly_P = self.res_anomaly.sel(lon=slice(self.sub_grid[0], self.sub_grid[1]), lat=slice(self.sub_grid[2], self.sub_grid[3]))
        
        # Calculate normal gravity at the ellipsoid
        self.LonP, LatP = np.meshgrid(self.res_anomaly_P['lon'], self.res_anomaly_P['lat'])  
        self.LatP = LatP
        self.gamma0 = normal_gravity_somigliana(phi=LatP, ellipsoid=self.ellipsoid)
        self.gamma0 *= 1e5
        
        
        self.Lon, self.Lat = np.meshgrid(self.res_anomaly['lon'], self.res_anomaly['lat'])
    
    def compute_geoid(self) -> np.ndarray:
        '''
        Compute the residual geoid height
        '''
        nrows_P, ncols_P = self.res_anomaly_P['Dg'].shape
        phip = np.radians(self.LatP)
        lonp = np.radians(self.LonP)
        coslonp = np.cos(lonp)
        sinlonp = np.sin(lonp)
        cosphip = np.cos(phip)
        sinphip = np.sin(phip)
        
        #### Near zone computation
        # cosphip = np.cos(phip)
        N_inner = earth()['radius'] / self.gamma0 * np.sqrt(cosphip * np.radians(self.dphi) * np.radians(self.dlam) * 1 / np.pi) * self.res_anomaly_P['Dg']
        
        self.inner = N_inner
        
        #### Far zone computation
        dn = np.round(self.ncols - ncols_P) + 1
        dm = np.round(self.nrows - nrows_P) + 1
        n1 = 0
        n2 = dn
        
        N_far = np.zeros_like(N_inner)
        
        for i in tqdm(range(nrows_P), desc='Computing far zone contribution'):
            m1 = 0
            m2 = dm
            for j in range(ncols_P):
                smallDg  = self.res_anomaly['Dg'].values[n1:n2, m1:m2]  
                smallphi = np.radians(self.Lat[n1:n2, m1:m2]) 
                smalllon = np.radians(self.Lon[n1:n2, m1:m2])  
                
                # Surface area on the sphere
                lat1 = smallphi - np.radians(self.dphi) / 2
                lat2 = smallphi + np.radians(self.dphi) / 2
                lon1 = smalllon - np.radians(self.dlam) / 2
                lon2 = smalllon + np.radians(self.dlam) / 2
                A_k  = earth()['radius']**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))

                cos_dlam = np.cos(smalllon) * coslonp[i,j] + np.sin(smalllon) * sinlonp[i,j]
                cos_psi  = sinphip[i,j] * np.sin(smallphi) + cosphip[i,j] * np.cos(smallphi) * cos_dlam
                sin2_psi_2 = np.sin( (phip[i,j] - smallphi) / 2 ) ** 2 + np.sin( (lonp[i,j] - smalllon) / 2 ) ** 2 * cosphip[i,j] * np.cos(smallphi)

                ### Stokes' function
                # Set S_k to zero when np.log(np.sqrt(sin2_psi_2) + sin2_psi_2) leads to NaN or 0 values 
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    log_arg = np.sqrt(sin2_psi_2) + sin2_psi_2
                    S_k = np.where(
                        log_arg <=0, 
                        0,
                        1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - 3 * cos_psi * np.log(log_arg)
                    )
                # S_k = 1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - 3 * cos_psi * np.log(np.sqrt(sin2_psi_2) + sin2_psi_2)

                # Spherical distance
                sd = haversine(np.degrees(lonp[i, j]), np.degrees(phip[i, j]), np.degrees(smalllon), np.degrees(smallphi), unit='deg')
                # Mask points outside the spherical cap
                sd[sd > self.sph_cap] = np.nan
                # Find the index of all NaN values in sd
                mask = np.isnan(sd)
                S_k[mask] = np.nan
                
                # print(S_k)
                c_k = A_k * S_k
                
                N_far[i, j] = np.nansum(c_k * smallDg) * 1 / (4 * np.pi * self.gamma0[i, j] * earth()['radius'])
                
                m1 += 1
                m2 += 1
            n1 += 1
            n2 += 1
            self.inner = N_inner
            self.outer = N_far
        
        return N_inner + N_far
        
        
        
    
    def stokes_function(self, psi: np.ndarray) -> np.ndarray:
        '''
        Compute Stokes' function matching MATLAB's implementation in Stokes_small.m.

        Parameters
        ----------
        psi       : Spherical distance in radians

        Returns
        -------
        S_k       : Stokes' function values
        '''
        # Compute auxiliary quantity: sin^2(psi/2)
        sin2_psi_2 = np.sin(psi / 2)**2  # Matches MATLAB's sin((...)/2).^2
        
        # Compute cos(psi) for consistency with MATLAB
        cos_psi = np.cos(psi)
        
        # Stokes' function as per MATLAB
        S_k = (1.0 / np.sqrt(sin2_psi_2) - 
            6.0 * np.sqrt(sin2_psi_2) + 
            1.0 - 
            5.0 * cos_psi - 
            3.0 * cos_psi * np.log(np.sqrt(sin2_psi_2) + sin2_psi_2))
        
        # Handle singularities (Inf to NaN, as in MATLAB)
        S_k[np.isinf(S_k)] = np.nan
        
        return S_k
    
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