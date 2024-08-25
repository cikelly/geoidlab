############################################################
# Utilities for modeling terrain quantities                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from . import constants
from easy_pygeoid.coordinates import geodetic2cartesian

import numpy as np
import xarray as xr
import rioxarray as rxr

class TerrainQuantities:
    '''
    Compute terrain quantities for use in geoid modeling
        - terrain correction
        - residual terrain modeling (RTM)
        - indirect effect
    '''
    def __init__(
        self, 
        Lon: float, 
        Lat: float,
        ref_topo: float=None, 
        sim_topo: float=None, 
        radius: float=100.,
        sub_grid=None,
        ellipsoid: str='wgs84'
    ) -> None:
        '''
        Initialize the TerrainQuantities class for terrain modeling
        
        Parameters
        ----------
        ref_topo  : (2D array) Reference topography
        sim_topo  : (2D array) Simulated (smooth) topography
        Lon, Lat  : (2D array) Longitude and latitude of ref_topo and sim_topo
        radius    : Integration radius in kilometers
        sub_grid  : (4-tuple) Subgrid over which to compute terrain effects
                        - In the form W, E, S, N
        ellipsoid : Reference ellipsoid

        Returns
        -------
        None
        
        Reference
        ---------
        Wichiencharoen (1982): The Indirect Effects On The Computation of Geoid Undulations
        https://ntrs.nasa.gov/citations/19830016735
        
        Notes
        -----
        1. ref_topo is the reference topography from a digital elevation model
        2. sim_topo is the smoothed topography at the same resolution as the GGM (e.g., DTM2006.0)
        '''
        self.R                 = constants.earth()['radius']
        self.rho               = constants.earth()['rho']
        self.G                 = constants.earth()['G']
        self.ref_topo          = ref_topo
        self.sim_topo          = sim_topo
        # self.ellipsoid         = constants.grs80() if ellipsoid == 'grs80' else constants.wgs84()
        self.ellipsoid         = ellipsoid
        self.sub_grid          = sub_grid
        self.nrows, self.ncols = ref_topo.shape
        self.radius            = radius * 1000 # meters
        
        if self.ref_topo is None and self.sim_topo is None:
            raise ValueError('Either ref_topo and sim_topo must be provided')
        
        # Define sub-grid
        if self.sub_grid is None:
            print(f'Defining sub-grid based on integration radius: {radius} km')
            self.radius_deg = self.km2deg((self.radius / 1000))
            min_lat = np.round(np.min(Lat) + self.radius_deg)
            max_lat = np.round(np.max(Lat) - self.radius_deg)
            min_lon = np.round(np.min(Lon) + self.radius_deg)
            max_lon = np.round(np.max(Lon) - self.radius_deg)
            self.sub_grid = (min_lon, max_lon, min_lat, max_lat)
        
        # Set ocean areas to zero
        if self.sim_topo is not None:
            self.sim_topo[self.sim_topo < 0] = 0
        if self.ref_topo is not None:
            self.ref_topo[self.ref_topo < 0] = 0

        
        
    @staticmethod
    def km2deg(km:float, radius:float=6371.) -> float:
        '''
        Convert kilometers to degrees
        
        Parameters
        ----------
        km        : kilometers
        radius    : radius of the sphere [default: earth radius (km)]
        
        Returns
        -------
        deg       : degrees
        
        Notes
        -----
        1. Using the radius of the sphere is more accurate than 2.
        2. km / 111.11 is a reasonable approximation, and works well in practice.
        3. The approach used here is the same as MATLAB's km2deg function
        '''
        rad = km / radius
        deg = rad * 180 / np.pi
        # km / 111.11
        return deg
    
    def terrain_correction(self):
        pass
    
    def rtm_anomaly(self):
        pass
    
    def rtm_zeta(self):
        pass