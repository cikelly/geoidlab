############################################################
# Utilities for modeling terrain quantities                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from . import constants
from easy_pygeoid.coordinates import geodetic2cartesian

class TerrainQuantities:
    '''
    '''
    def __init__(
        self, 
        ref_topo, 
        sim_topo, 
        Lon, Lat,
        sub_grid=[0,0,0,0],
        ellipsoid=constants.wgs84()
    ) -> None:
        '''
        Initialize the TerrainQuantities class for terrain modeling
        
        Parameters
        ----------
        ref_topo  : (2D array) Reference topography
        sim_topo  : (2D array) Simulated (smooth) topography
        Lon, Lat  : (2D array) Longitude and latitude of ref_topo and sim_topo
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
        
        min_lon, max_lon, min_lat, max_lat = sub_grid
        
        self.sim_topo[self.sim_topo < 0] = 0
    
    def terrain_correction(self):
        pass
    
    def rtm_anomaly(self):
        pass
    
    def rtm_zeta(self):
        pass