############################################################
# Utilities for Height System Unification                  #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union



from geoidlab.gravity import normal_gravity, normal_gravity_above_ellipsoid as normalGravityBM
from geoidlab.ggm import GlobalGeopotentialModel

import numpy as np
from pyproj import Geod

def find_nearest_neighbors_ellipsoidal(data, lat_col='lat', lon_col='lon', ellipsoid='grs80') -> np.ndarray:
    '''
    Compute nearest neighbors using ellipsoidal geodesic distances (Vincenty-like).
    '''
    
    n = len(data)
    dist = np.full((n, n), np.inf)
    geod = Geod(ellps=ellipsoid.upper())  # 'grs80' or 'wgs84'
    
    lats = data[lat_col].values
    lons = data[lon_col].values
    
    for i in range(n):
        for j in range(i + 1, n):
            _, _, d = geod.inv(lons[i], lats[i], lons[j], lats[j])  # Returns azimuths and distance (m)
            dist[i, j] = d
            dist[j, i] = d
    
    min_indices = np.argmin(dist, axis=1)
    return min_indices

class HeightSystemUnifier:
    '''
    Class for height system unification including applying corrections
    to obtain Helmert orthometric heights, normal heights, and normal-orthometric heights.
    from measured height differences.
    
    References
    ----------
    1. Physical Geodesy by Hofmann-Wellenhof & Moritz (2006): Chapter 4
    '''
    
    def __init__(
        self, 
        normal_gravity_above_ellipsoid=None,
        disturbing_potential=None,
        ellipsoid: str = 'wgs84',
        data: Union[Path, str, pd.DataFrame] = None,
        ggm_model=None,
        ggm_nmax: int = 90,
        model_dir: str = 'downloads',
        chunk_size: int = 200,
        has_gravity: bool = False,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        elev_col: str = 'elevation',
        ellip_height_col: str = 'h',
        gravity_col: str = 'gravity'
    ) -> None:
        '''
        Initialize HeightSystemUnifier
        
        Parameters
        ----------
        normal_gravity_above_ellipsoid: Normal gravity above the ellipsoid
        disturbing_potential          : Disturbing potential
        ellipsoid                     : Reference ellipsoid
        data                          : Benchmark data
                                        Must contain columns: lon, lat, elevation, ellipsoidal height, and optionally, gravity
                                        Units: deg, deg, meters, meters, mGal
        '''
        if data is None:
            raise ValueError('Data containing lon, lat, elevation, and optionally gravity is required')
        
        if isinstance(data, str) or isinstance(data, Path):
            data = pd.read_csv(data)
        
        required_cols = [lon_col, lat_col, elev_col]
        if has_gravity:
            required_cols.append(gravity_col)
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise KeyError(f'Missing columns: {missing}')        
        
        self.normal_gravity_above_ellipsoid = normal_gravity_above_ellipsoid
        self.disturbing_potential = disturbing_potential
        self.ellipsoid = ellipsoid
        self.data = data
        self.ggm_model = ggm_model
        self.nmax = ggm_nmax
        self.model_dir = model_dir
        self.chunk_size = chunk_size
        self.has_gravity = has_gravity
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.elev_col = elev_col
        self.gravity_col = gravity_col
        self.ellip_height_col = ellip_height_col
        
        self.gamma_45 = normal_gravity(phi=45, ellipsoid=ellipsoid) * 1e2 # m/s2 to gal
        
        # Calculate distances between points and get the indices of minimum distance from each point
        self.min_indices = find_nearest_neighbors_ellipsoidal(self.data, lat_col=self.lat_col, lon_col=self.lon_col, ellipsoid=self.ellipsoid)
        
    
    def helmert_correction(
        self,  
        parallel: bool = True,
        batch_size: int = 1000,
        gravity_disturbance = None
    ) -> np.ndarray:
        '''
        Compute Helmert orthometric correction for a given height difference.
        
        Parameters
        ----------
        parallel
        gravity_disturbance: Gravity disturbance in mGal
        
        Returns
        -------
        helmert_correction: Helmert correction (m)
        
        References
        ----------
        1. Physical Geodesy by Hofmann-Wellenhof & Moritz (2006): p. 165
        '''
        
        H_original = np.asarray(self.data[self.elev_col], dtype=float)
        H = H_original * 1e-3
        lon = np.asarray(self.data[self.lon_col], dtype=float)
        lat = np.asarray(self.data[self.lat_col], dtype=float)
        h = np.asarray(self.data[self.ellip_height_col], dtype=float)
        
        if len(lat) > 1 and np.all(self.min_indices == np.arange(len(self.min_indices))):
            print('Warning: All points appear coincident or no valid neighbors found; corrections set to 0.')
            helmert_corrections = np.zeros(len(lat))
            corrected_heights = H_original + helmert_corrections
            return helmert_corrections, corrected_heights
        
        data = pd.DataFrame({'lon': lon, 'lat': lat, 'elevation': h})

        if self.has_gravity:
            # g = self.data[self.gravity_col]
            # g = g * 1e-3
            g = np.asarray(self.data[self.gravity_col], dtype=float) * 1e-3
            
        else:
            if self.ggm_model is None and gravity_disturbance:
                raise ValueError('Provide gravity disturbance for data points or specify a GGM to enable synthesis.')
            elif gravity_disturbance:
                # print('Gravity disturbance provided and will be used.')
                print('Using provided gravity disturbance.')
                # dg_p = gravity_disturbance
                dg_p = np.asarray(gravity_disturbance, dtype=float)
            else:
                print(f'Data does not contain gravity observations and gravity disturbance was not provided. \nCalculating gravity disturbance from {self.ggm_model} ...')
                ggm = GlobalGeopotentialModel(
                    model_name=self.ggm_model,
                    grav_data=data,
                    ellipsoid=self.ellipsoid,
                    nmax=self.nmax,
                    zonal_harmonics=True,
                    model_dir=self.model_dir,
                    chunk_size=self.chunk_size
                )
                dg_p = ggm.gravity_disturbance(parallel=parallel, batch_size=batch_size)
            
                print('Computing normal gravity at data points ...')
                gamma_p = normalGravityBM(phi=np.asarray(self.data[self.lat_col], dtype=float), h=h, ellipsoid=self.ellipsoid)
                
                # Convert gravity quantities from mGal to Gal and height from meters to kilometers
                gamma_p = gamma_p * 1e-3 # mGal to Gal
                dg_p = dg_p * 1e-3 # mGal to Gal
                g = dg_p + gamma_p
        
        
        # Mean gravity
        g_mean = g + 0.0424 * H # (Prey reduction -- Physical Geodesy Eq. 4-31)
        
        # Update H, g, and g_mean_new to be arranged in terms of minimum distance
        Hnew = H[self.min_indices]
        gnew = g[self.min_indices]
        g_mean_new = g_mean[self.min_indices]
        
        # Compute vectorized Helmert corrections (in km)
        
        dH = H - Hnew
        A = dH / self.gamma_45 * (g + gnew - 2 * self.gamma_45)
        B = (g_mean - self.gamma_45) / self.gamma_45 * H
        C = (g_mean_new - self.gamma_45) / self.gamma_45 * Hnew
        HOC = A + B - C
        
        # Convert to meters
        helmert_corrections = HOC * 1e3
        
        corrected_heights = H_original + helmert_corrections
        
        return helmert_corrections, corrected_heights
        

    
    def normal_correction(self) -> np.ndarray:
        '''
        Compute normal correction for a given height difference.
        
        Returns
        -------
        normal_correction: Normal correction (m)
        
        References
        ----------
        1. Physical Geodesy by Hofmann-Wellenhof & Moritz (2006): p. 168
        '''
        
        
        
        return self.helmert_correction()
        