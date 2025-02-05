############################################################
# Utilities for modeling terrain quantities                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import numpy as np
import xarray as xr
import bottleneck as bn

import time
import sys
import threading

from . import constants
from .coordinates import geodetic2cartesian
from .utils.parallel_utils import compute_tc_chunk

from tqdm import tqdm
from joblib import Parallel, delayed

class TerrainQuantities:
    '''
    Compute terrain quantities for use in geoid modeling
        - terrain correction
        - residual terrain modeling (RTM)
        - indirect effect
    '''
    def __init__(
        self, 
        ori_topo: xr.Dataset, 
        ref_topo: xr.Dataset=None, 
        radius: float=110.,
        ellipsoid: str='wgs84',
        bbox_off: float=1.
    ) -> None:
        '''
        Initialize the TerrainQuantities class for terrain modeling

        Parameters
        ----------
        ori_topo  : (2D array) Original topography
        ref_topo  : (2D array) Reference (smooth) topography
        radius    : Integration radius in kilometers
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
        1. ori_topo is the original topography from a digital elevation model
        2. ref_topo is the smoothed topography at the same resolution as the GGM (e.g., DTM2006.0)
        3. sub_grid represents the study area (computation points)
        4. radius is the maximum distance beyond which cells are excluded from the TC computation
           from the computation point. Beyond this distance (usually 1 degree), the contribution of
           cells to the TC is considered negligible.
        '''
        self.R           = constants.earth()['radius']
        self.rho         = constants.earth()['rho']
        self.G           = constants.earth()['G']
        self.ellipsoid   = ellipsoid
        self.radius      = radius * 1000 # meters
        self.bbox_off    = bbox_off

        if ori_topo is None and ref_topo is None:
            raise ValueError('At least ori_topo must be provided')

        # Rename coordinates and data variables if necessary
        ori_topo = TerrainQuantities.rename_variables(ori_topo)
        ref_topo = TerrainQuantities.rename_variables(ref_topo) if ref_topo is not None else None

        self.ori_topo = ori_topo
        self.ref_topo = ref_topo
        self.nrows, self.ncols = self.ori_topo['z'].shape

        # Set ocean areas to zero
        if self.ref_topo is not None:
            self.ref_topo['z'] = self.ref_topo['z'].where(self.ref_topo['z'] >= 0, 0)
        self.ori_topo['z'] = self.ori_topo['z'].where(self.ori_topo['z'] >= 0, 0)

        # Define sub-grid and extract data
        lon = self.ori_topo['x'].values
        lat = self.ori_topo['y'].values
        # print(f'Defining sub-grid based on integration radius: {radius} km')
        self.radius_deg = self.km2deg((self.radius / 1000))
        min_lat = round(min(lat) + self.radius_deg)
        max_lat = round(max(lat) - self.radius_deg)
        min_lon = round(min(lon) + self.radius_deg)
        max_lon = round(max(lon) - self.radius_deg)
        
        # Ensure sub-grid is within bounds
        if min_lat >= min(lat) or max_lat >= max(lat) or min_lon >= min(lon) or max_lon <= max(lon):
            self.sub_grid = (
                min(lon)+self.bbox_off, 
                max(lon)-self.bbox_off, 
                min(lat)+self.bbox_off, 
                max(lat)-self.bbox_off
            )
        else:
            self.sub_grid = (min_lon, max_lon, min_lat, max_lat)
            
        # Extract sub-grid topography
        self.ori_P = self.ori_topo.sel(x=slice(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3]))
        self.ref_P = self.ref_topo.sel(x=slice(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3])) if self.ref_topo else None

        # Grid size in x and y
        self.dlam = (max(lon) - min(lon)) / (self.ncols - 1)
        self.dphi = (max(lat) - min(lat)) / (self.nrows - 1)
        dx = TerrainQuantities.deg2km(self.dlam) * 1000 # meters
        dy = TerrainQuantities.deg2km(self.dphi) * 1000 # meters
        
        # Precompute G_rho_dxdy
        self.G_rho_dxdy = self.G * self.rho * dx * dy
        
        # Get cartesian coordinates of the original topography (running point)
        Lon, Lat = np.meshgrid(lon, lat)
        _, self.X, self.Y, self.Z = geodetic2cartesian(phi=Lat.flatten(), lambd=Lon.flatten(), ellipsoid=self.ellipsoid)
        self.X = self.X.reshape(self.nrows, self.ncols)
        self.Y = self.Y.reshape(self.nrows, self.ncols)
        self.Z = self.Z.reshape(self.nrows, self.ncols)

        # Get cartesian coordinates of the sub-grid (computation points)
        LonP, LatP = np.meshgrid(self.ori_P['x'].values, self.ori_P['y'].values)
        _, self.Xp, self.Yp, self.Zp = geodetic2cartesian(phi=LatP.flatten(), lambd=LonP.flatten(), ellipsoid=self.ellipsoid)
        self.Xp = self.Xp.reshape(LonP.shape)
        self.Yp = self.Yp.reshape(LonP.shape)
        self.Zp = self.Zp.reshape(LonP.shape)
        self.LonP = LonP
        self.LatP = LatP

        lamp = np.radians(self.ori_P['x'].values)
        phip = np.radians(self.ori_P['y'].values)
        lamp, phip = np.meshgrid(lamp, phip)
        
        self.coslamp = np.cos(lamp)
        self.sinlamp = np.sin(lamp)
        self.cosphip = np.cos(phip)
        self.sinphip = np.sin(phip)

    def terrain_correction_sequential(self) -> np.ndarray:
        '''
        Compute terrain correction

        Returns
        -------
        tc     : Terrain Correction
        '''
        nrows_P, ncols_P = self.ori_P['z'].shape
        tc = np.zeros((nrows_P, ncols_P))
        dn = np.round(self.ncols - ncols_P) + 1
        dm = np.round(self.nrows - nrows_P) + 1

        n1 = 1
        n2 = dn

        Hp   = self.ori_P['z'].values 

        for i in tqdm(range(nrows_P), desc='Computing terrain correction'):
            m1 = 1
            m2 = dm
            for j in range(ncols_P):
                smallH = self.ori_topo['z'].values[n1:n2, m1:m2]
                smallX = self.X[n1:n2, m1:m2]
                smallY = self.Y[n1:n2, m1:m2]
                smallZ = self.Z[n1:n2, m1:m2]

                # Local coordinates (x, y)
                x = self.coslamp[i, j] * (smallY - self.Yp[i, j]) - \
                    self.sinlamp[i, j] * (smallX - self.Xp[i, j])
                y = self.cosphip[i, j] * (smallZ - self.Zp[i, j]) - \
                    self.coslamp[i, j] * self.sinphip[i, j] * (smallX - self.Xp[i, j]) - \
                    self.sinlamp[i, j] * self.sinphip[i, j] * (smallY - self.Yp[i, j])

                # Distances
                d = np.hypot(x, y)
                # d = np.where(d <= self.radius, d, np.nan)
                d[d > self.radius] = np.nan
                d3 = d * d * d
                d5 = d3 * d * d
                d7 = d5 * d * d

                # Integrate the terrain correction
                DH2 = (smallH - Hp[i, j]) ** 2 #* (smallH - Hp[i, j])
                DH4 = DH2 * DH2
                DH6 = DH4 * DH2
                c1  = 0.5 *  self.G_rho_dxdy * bn.nansum(DH2 / d3)      # 1/2
                c2  = -0.375 * self.G_rho_dxdy * bn.nansum(DH4 / d5)    # 3/8
                c3  = 0.3125 * self.G_rho_dxdy * bn.nansum(DH6 / d7)    # 5/16
                tc[i, j] = (c1 + c2 + c3) * 1e5 # [mGal]
                # moving window
                m1 += 1
                m2 += 1
            n1 += 1
            n2 += 1
        return tc

    def terrain_correction_parallel(
        self, 
        chunk_size: int=10, 
        progress=True, 
    ) -> np.ndarray:
        '''
        Compute terrain correction (parallelized with chunking).
        
        Parameters
        ----------
        chunk_size : number of rows to process in each chunk
        progress   : Progress bar display
        
        Returns
        -------
        tc         : Terrain Correction
        or
        Dg_RTM     : Residual terrain (RTM) gravity anomalies
        '''
        if progress:
            def print_progress(stop_signal) -> None:
                '''
                Prints '#' every second to indicate progress.
                '''
                while not stop_signal.is_set():
                    sys.stdout.write("#")
                    sys.stdout.flush()
                    time.sleep(1.5)  # Adjust the frequency as needed

        nrows_P, ncols_P = self.ori_P['z'].shape
        tc = np.zeros((nrows_P, ncols_P))
        dn = np.round(self.ncols - ncols_P) + 1
        dm = np.round(self.nrows - nrows_P) + 1

        Hp = self.ori_P['z'].values

        # Divide rows into chunks
        chunks = [
            (i, min(i + chunk_size, nrows_P)) 
            for i in range(0, nrows_P, chunk_size)
        ]

        print('Computing terrain correction...') 
        
        if progress:
            stop_signal = threading.Event()
            progress_thread = threading.Thread(target=print_progress, args=(stop_signal,))
            progress_thread.start()

        # Submit tasks for each chunk

        results = Parallel(n_jobs=-1)(
            delayed(compute_tc_chunk)(
                row_start, row_end, ncols_P, dm, dn, self.coslamp, self.sinlamp, self.cosphip, 
                self.sinphip, Hp, self.ori_topo['z'].values, self.X, self.Y, self.Z, self.Xp, 
                self.Yp, self.Zp, self.radius, self.G_rho_dxdy
            ) for row_start, row_end in chunks
        )
        
        if progress:
            stop_signal.set()
            progress_thread.join()
            print('\nCompleted.')
        
        # Collect results
        for row_start, row_end, tc_chunk in results:
            tc[row_start:row_end, :] = tc_chunk
        return tc

    
    def terrain_correction(
        self,
        parallel: bool=True,
        chunk_size: int=10,
        progress: bool=True,
    ) -> np.ndarray:
        '''
        Compute terrain correction.

        Parameters
        ----------
        parallel   : True/False
                    If True, use the parallelized version. Default: True.
        chunk_size : int
                    Size of the chunk in terms of number of rows. Default is 10.
        progress   : True/False
                    If True, display a progress bar. Default: True.
        
        Return
        ------
        tc       : Terrain Correction
        '''
        if parallel:
            return self.terrain_correction_parallel(chunk_size=chunk_size, progress=progress)
        else:
            return self.terrain_correction_sequential()
        # return self.terrain_correction_parallel(chunk_size=chunk_size, progress=progress) if parallel else self.terrain_correction_sequential()

    
    def rtm_anomaly(
        self, 
        parallel: bool=True, 
        chunk_size: int=10, 
        progress: bool=True
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute residual terrain (RTM) gravity anomalies
        
        Parameters
        ----------
        parallel  : Terrain Correction
        chunk_size: int
                    Size of the chunk in terms of number of rows. Default is 10.
        progress  : True/False
                    If True, display a progress bar. Default: True.
        
        Returns
        -------
        Dg_RTM   : Residual terrain (RTM) gravity anomalies [mgal]
        
        Reference
        ---------
        1. Forsberg & Tscherning (1984): Topographic effects in gravity field modelling for BVP
           Equation 19
        
        Notes
        -----
        1. Dg_RTM has the same equation as terrain correction, with the difference being that
           the residual terrain (H - H_ref) is used instead of the original terrain
        '''
        
        if parallel:
            return self.terrain_correction_parallel(chunk_size=chunk_size, progress=progress, effect='rtm')
        else:
            return self.terrain_correction_sequential(effect='rtm')


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
    