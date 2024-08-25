############################################################
# Utilities for modeling terrain quantities                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from . import constants
from easy_pygeoid.coordinates import geodetic2cartesian

class TerrainQuantities:
    '''
    Compute terrain quantities for use in geoid modeling
        - terrain correction
        - residual terrain modeling (RTM)
        - indirect effect
    '''
    def __init__(
        self, 
        ref_topo: float, 
        sim_topo: float, 
        Lon:float, 
        Lat:float,
        radius:float=100.,
        sub_grid=None,
        ellipsoid:str='wgs84'
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
        self.ellipsoid         = ellipsoid
        self.sub_grid          = sub_grid
        self.nrows, self.ncols = ref_topo.shape
        self.radius            = radius * 1000 # meters
        
        # Define sub-grid
        if self.sub_grid is None:
            self.radius_deg = self.km2deg((self.radius / 1000))
        
        
        self.rows, self.cols = self.ref_topo.shape
        min_lon, max_lon, min_lat, max_lat = sub_grid
        
        self.sim_topo[self.sim_topo < 0] = 0
    
    @staticmethod
    def km2deg(km:float) -> float:
        '''
        Convert kilometers to degrees
        '''
        return km / 111.11
    
    def terrain_correction(self):
        pass
    
    def rtm_anomaly(self):
        pass
    
    def rtm_zeta(self):
        pass