############################################################
# Utilities for modeling terrain quantities                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from . import constants
from easy_pygeoid.coordinates import geodetic2cartesian
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import rioxarray as rxr
import bottleneck as bn

class TerrainQuantities:
    '''
    Compute terrain quantities for use in geoid modeling
        - terrain correction
        - residual terrain modeling (RTM)
        - indirect effect
    '''
    def __init__(
        self, 
        ref_topo: xr.Dataset, 
        sim_topo: xr.Dataset=None, 
        radius: float=100.,
        sub_grid=None,
        ellipsoid: str='wgs84',
        bbox_off: float=1.
    ) -> None:
        '''
        Initialize the TerrainQuantities class for terrain modeling
        
        Parameters
        ----------
        ref_topo  : (2D array) Reference topography
        sim_topo  : (2D array) Simulated (smooth) topography
        radius    : Integration radius in kilometers
        sub_grid  : (4-tuple) Subgrid over which to compute terrain effects
                        - In the form W, E, S, N
        ellipsoid : Reference ellipsoid
        bbox_off  : Offset in degrees for bounding box

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
        # self.ellipsoid         = constants.grs80() if ellipsoid == 'grs80' else constants.wgs84()
        self.ellipsoid         = ellipsoid
        self.sub_grid          = sub_grid
        self.radius            = radius * 1000 # meters
        self.bbox_off          = bbox_off
        
        if ref_topo is None and sim_topo is None:
            raise ValueError('Either ref_topo and sim_topo must be provided')
        
        # Rename coordinates and data variables if necessary
        ref_topo = TerrainQuantities.rename_variables(ref_topo)
        sim_topo = TerrainQuantities.rename_variables(sim_topo) if sim_topo is not None else None
        
        self.ref_topo = ref_topo
        self.sim_topo = sim_topo
        self.nrows, self.ncols = self.ref_topo['z'].shape
        
        # Set ocean areas to zero
        if self.sim_topo is not None:
            self.sim_topo['z'] = self.sim_topo['z'].where(self.sim_topo['z'] >= 0, 0)
        self.ref_topo['z'] = self.ref_topo['z'].where(self.ref_topo['z'] >= 0, 0)

        # Define sub-grid and extract data
        if self.sub_grid is None:
            lon = self.ref_topo['x'].values
            lat = self.ref_topo['y'].values
            print(f'Defining sub-grid based on integration radius: {radius} km')
            self.radius_deg = self.km2deg((self.radius / 1000))
            min_lat = round(min(lat) + self.radius_deg)
            max_lat = round(max(lat) - self.radius_deg)
            min_lon = round(min(lon) + self.radius_deg)
            max_lon = round(max(lon) - self.radius_deg)
            
            # Check if any of the conditions are not met
            if min_lat >= min(lat) or max_lat >= max(lat) or min_lon >= min(lon) or max_lon <= max(lon):
                self.sub_grid = (
                    min(lon)+self.bbox_off, 
                    max(lon)-self.bbox_off, 
                    min(lat)+self.bbox_off, 
                    max(lat)-self.bbox_off
                )
            else:
                self.sub_grid = (min_lon, max_lon, min_lat, max_lat)
                
            
        self.ref_P = self.ref_topo.sel(x=slice(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3]))
        self.sim_P = self.sim_topo.sel(x=slice(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3])) if self.sim_topo else None

        # Grid size in x and y
        self.dlam = (max(lon) - min(lon)) / (self.ncols - 1)
        self.dphi = (max(lat) - min(lat)) / (self.nrows - 1)
        self.dx = TerrainQuantities.deg2km(self.dlam) * 1000 # meters
        self.dy = TerrainQuantities.deg2km(self.dphi) * 1000 # meters
        
        # Get cartesian coordinates of the original and subg-rid
        Lon, Lat = np.meshgrid(lon, lat)
        _, self.X, self.Y, self.Z = geodetic2cartesian(phi=Lat.flatten(), lambd=Lon.flatten(), ellipsoid=self.ellipsoid)
        self.X = self.X.reshape(self.nrows, self.ncols)
        self.Y = self.Y.reshape(self.nrows, self.ncols)
        self.Z = self.Z.reshape(self.nrows, self.ncols)
        
        LonP, LatP = np.meshgrid(self.ref_P['x'].values, self.ref_P['y'].values)
        _, self.Xp, self.Yp, self.Zp = geodetic2cartesian(phi=LatP.flatten(), lambd=LonP.flatten(), ellipsoid=self.ellipsoid)
        self.Xp = self.Xp.reshape(LonP.shape)
        self.Yp = self.Yp.reshape(LonP.shape)
        self.Zp = self.Zp.reshape(LonP.shape)
        self.LonP = LonP
        self.LatP = LatP

    def compute_terrain_correction(self, i, j, n1, n2, m1, m2) -> tuple:
        # Extract small grids
        smallH = self.ref_topo['z'].values[n1:n2, m1:m2]
        smallX = self.X[n1:n2, m1:m2]
        smallY = self.Y[n1:n2, m1:m2]
        smallZ = self.Z[n1:n2, m1:m2]

        # Precompute radians
        lon_rad = np.radians(self.LonP[i, j])
        lat_rad = np.radians(self.LatP[i, j])

        # Local coordinates (x, y)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        x = cos_lon * (smallY - self.Yp[i, j]) - \
            sin_lon * (smallX - self.Xp[i, j])
        y = cos_lat * (smallZ - self.Zp[i, j]) - \
            cos_lon * sin_lat * (smallX - self.Xp[i, j]) - \
            sin_lon * sin_lat * (smallY - self.Yp[i, j])

        # Distances
        d = np.hypot(x, y)
        # d_masked = np.where(d <= self.radius, d, np.nan)
        d[d > self.radius] = np.nan 
        d3 = d * d * d
        d5 = d3 * d * d
        d7 = d5 * d * d
        
        # Terrain effect in mGal
        DH2 = (smallH - self.ref_P['z'].values[i, j]) ** 2
        dxdy = self.dx * self.dy
        G_rho_dxdy = self.G * self.rho * dxdy
        
        c1 = 1/2 * G_rho_dxdy * bn.nansum(DH2 / d3)
        c2 = -3/8 * G_rho_dxdy * bn.nansum((DH2 * DH2) / d5)
        c3 = 5/16 * G_rho_dxdy * bn.nansum((DH2 * DH2 * DH2) / d7)
        
        result = (c1 + c2 + c3) * 1e5  # mGal
        return i, j, result

    def terrain_correction(self, progress=True, batch_size=20) -> np.ndarray:
        '''
        Compute terrain correction
        
        Returns
        -------
        tc    : terrain correction
        '''
        nrows, ncols = self.Xp.shape
        dm = round(self.nrows - nrows) + 1
        dn = round(self.ncols - ncols) + 1
        tc = np.zeros((nrows, ncols))

        def process_batch(start_row, end_row) -> list:
            results = []
            for i in range(start_row, end_row):
                n1 = i
                n2 = i + dn
                m1 = 0
                m2 = dm
                row_results = np.zeros(ncols)
                for j in range(ncols):
                    _, _, result = self.compute_terrain_correction(i, j, n1, n2, m1, m2)
                    row_results[j] = result
                    m1 += 1
                    m2 += 1
                results.append((i, row_results))
            return results

        with ThreadPoolExecutor() as executor:
            futures = []
            if progress:
                for start_row in tqdm(range(0, nrows, batch_size), desc='Submitting TC tasks to workers'):
                    end_row = min(start_row + batch_size, nrows)
                    futures.append(executor.submit(process_batch, start_row, end_row))
            else:
                print('Submitting TC tasks to workers')
                for start_row in range(0, nrows, batch_size):
                    end_row = min(start_row + batch_size, nrows)
                    futures.append(executor.submit(process_batch, start_row, end_row))

            if progress:
                for future in tqdm(futures, desc='Retrieving TC from workers'):
                    batch_results = future.result()
                    for i, row_results in batch_results:
                        tc[i, :] = row_results
            else:
                print('Retrieving TC from workers...')
                for future in futures:
                    batch_results = future.result()
                    for i, row_results in batch_results:
                        tc[i, :] = row_results

        return tc

    def rtm_anomaly(self) -> np.ndarray:
        pass
    
    def rtm_zeta(self) -> np.ndarray:
        pass
    
    @staticmethod
    def rename_variables(ds) -> xr.Dataset:
        coord_names = {
            'x': ['lon'],
            'y': ['lat'],
            'z': ['elevation', 'elev', 'height', 'h', 'dem']
        }
        
        rename_dict = {}
        
        for name in ds.coords.keys() | ds.data_vars.keys():
            lower_name = name.lower()
            for standard_name, possible_names in coord_names.items():
                if any(possible_name in lower_name for possible_name in possible_names):
                    rename_dict[name] = standard_name
                    break
        
        return ds.rename(rename_dict)
            
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
    
    @staticmethod
    def deg2km(deg, radius=6371.) -> float:
        '''
        Convert degrees to kilometers
        
        Parameters
        ----------
        deg       : degrees
        radius    : radius of the sphere [default: earth radius (km)]
        
        Returns
        -------
        km        : kilometers
        
        Notes
        -----
        1. Using the radius of the sphere is more accurate than 2.
        2. deg * 111.11 is a reasonable approximation, and works well in practice.
        3. The approach used here is the same as MATLAB's deg2km function
        '''
        rad = deg * np.pi / 180
        km = rad * radius
        
        return km
    


