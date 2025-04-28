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

from geoidlab import constants
from geoidlab.coordinates import geodetic2cartesian
from geoidlab.utils.parallel_utils import (
    compute_tc_chunk, 
    compute_rtm_tc_chunk, 
    compute_ind_chunk, 
    compute_rtm_height_anomaly_chunk
)
from geoidlab.gravity import normal_gravity_somigliana
from geoidlab.utils.io import save_to_netcdf

from tqdm import tqdm
from joblib import Parallel, delayed

class TerrainQuantities:
    '''
    Compute terrain quantities for use in geoid modeling
        - terrain correction
        - residual terrain modeling (RTM)
        - indirect effect
    '''
    VALID_WINDOW_MODES = {'small', 'radius'}
    
    def __init__(
        self, 
        ori_topo: xr.Dataset, 
        ref_topo: xr.Dataset = None, 
        radius: float = 110.,
        ellipsoid: str = 'wgs84',
        bbox_off: float = 1.,
        sub_grid: tuple[float, float, float, float] = None,
        proj_dir: str = None,
        window_mode: str = 'small',
        overwrite: bool = True,
    ) -> None:
        '''
        Initialize the TerrainQuantities class for terrain modeling

        Parameters
        ----------
        ori_topo   : (2D array) Original topography
        ref_topo   : (2D array) Reference (smooth) topography
        radius     : Integration radius in kilometers
        ellipsoid  : Reference ellipsoid
        bbox_off   : Offset in degrees for bounding box
        sub_grid   : Bounding coordinates of the area of interest
        proj_dir   : Directory to save the output
        window_mode: 'small' or 'radius'
                    - 'small' uses a small window of size
                    - 'radius' uses a radius-based window
        overwrite  : Overwrite existing files when saving

        Returns
        -------
        None

        Reference
        ---------
        1. Wichiencharoen (1982): The Indirect Effects On The Computation of Geoid Undulations
           https://ntrs.nasa.gov/citations/19830016735
        2. Forsberg & Tscherning (1984): Topographic effects in gravity field modelling for BVP
           Equation 19

        Notes
        -----
        1. ori_topo is the original topography from a digital elevation model
        2. ref_topo is the smoothed topography at the same resolution as the GGM (e.g., DTM2006.0)
        3. sub_grid represents the study area (computation points)
        4. radius is the maximum distance beyond which cells are excluded from the TC computation
           from the computation point. Beyond this distance (usually 1 degree), the contribution of
           cells to the TC is considered negligible.
        '''
        self.R             = constants.earth()['radius']
        self.rho           = constants.earth()['rho']
        self.G             = constants.earth()['G']
        self.ellipsoid     = ellipsoid
        self.radius        = radius * 1000 # meters
        self.bbox_off      = bbox_off
        self.proj_dir      = proj_dir
        self.window_mode   = window_mode.lower()
        self.overwrite     = overwrite
        
        # Validate window mode
        if self.window_mode not in self.VALID_WINDOW_MODES:
            raise ValueError(f'Invalid window mode: {self.window_mode}. Must be one of {sorted(self.VALID_WINDOW_MODES)}.')

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
        if sub_grid is None:
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
        else:
            self.sub_grid = sub_grid

        # Extract sub-grid topography
        self.ori_P = self.ori_topo.sel(x=slice(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3]))
        self.ref_P = self.ref_topo.sel(x=slice(self.sub_grid[0], self.sub_grid[1]), y=slice(self.sub_grid[2], self.sub_grid[3])) if self.ref_topo else None

        # Grid size in x and y
        self.dlam = (max(lon) - min(lon)) / (self.ncols - 1)
        self.dphi = (max(lat) - min(lat)) / (self.nrows - 1)
        dx = TerrainQuantities.deg2km(self.dlam) * 1000 # meters
        dy = TerrainQuantities.deg2km(self.dphi) * 1000 # meters
        
        # Precompute G_rho_dxdy and 2 * pi * G * rho
        self.G_rho_dxdy = self.G * self.rho * dx * dy
        self.two_pi_G_rho = 2 * np.pi * self.G * self.rho
        
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
        
        # Precompute window sizes
        if self.window_mode == 'small':
            self.dn = int(np.round(self.ncols - self.ori_P['z'].shape[1])) + 1
            self.dm = int(np.round(self.nrows - self.ori_P['z'].shape[0])) + 1
        else:
            self.dn = int(np.ceil(self.radius_deg / self.dlam)) * 2 + 1
            self.dm = int(np.ceil(self.radius_deg / self.dphi)) * 2 + 1
            self.dn = min(self.dn, self.ncols)
            self.dm = min(self.dm, self.nrows)

    def get_window(
        self,
        i: int,
        j: int,
        # window_mode: str = 'small',
        include_ref: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Compute the windowed data for a given computation point
        
        Parameters
        ----------
        i           : Row index of the computation point
        j           : Column index of the computation point
        window_mode : 'small' or 'radius'
                        'small' uses a small window of size 
                        'radius' uses a radius-based window
        include_ref : Include reference topography for RTM

        Returns
        -------
        smallH      : Windowed topography
        smallX      : Windowed X coordinates
        smallY      : Windowed Y coordinates
        smallZ      : Windowed Z coordinates
        smallH_ref  : Windowed reference topography (if include_ref is True)
        '''
        # if window_mode not in self.VALID_WINDOW_MODES:
        #     raise ValueError(f'Invalid window mode: {window_mode}. Must be one of {sorted(self.VALID_WINDOW_MODES)}.')

        # if window_mode == 'small':
        #     dn = int(np.round(self.ncols - self.ori_P['z'].shape[1])) + 1
        #     dm = int(np.round(self.nrows - self.ori_P['z'].shape[0])) + 1
        # else:
        #     dn = int(np.ceil(self.radius_deg / self.dlam)) * 2 + 1
        #     dm = int(np.ceil(self.radius_deg / self.dphi)) * 2 + 1
        #     dn = min(dn, self.ncols)
        #     dm = min(dm, self.nrows)
        
        i_center = int(np.round((self.LatP[i, j] - self.ori_topo['y'].values[0]) / self.dphi))
        j_center = int(np.round((self.LonP[i, j] - self.ori_topo['x'].values[0]) / self.dlam))
        i_start = max(0, i_center - self.dm // 2)
        i_end = min(self.nrows, i_start + self.dm)
        j_start = max(0, j_center - self.dn // 2)
        j_end = min(self.ncols, j_start + self.dn)
        i_start = max(0, i_end - self.dm)
        j_start = max(0, j_end - self.dn)
        # i_start = i_end - self.dm
        # j_start = j_end - self.dn

        smallH = self.ori_topo['z'].values[i_start:i_end, j_start:j_end]
        smallX = self.X[i_start:i_end, j_start:j_end]
        smallY = self.Y[i_start:i_end, j_start:j_end]
        smallZ = self.Z[i_start:i_end, j_start:j_end]
        
        if include_ref and self.ref_topo is not None:
            smallH_ref = self.ref_topo['z'].values[i_start:i_end, j_start:j_end]
            return smallH, smallX, smallY, smallZ, smallH_ref
        
        return smallH, smallX, smallY, smallZ
    
    def get_window_indices(
        self, 
        i: int, 
        j: int, 
        # window_mode: str = 'small'
    ) -> tuple[int, int, int, int]:
        '''
        Compute the window indices for parallel processing
        '''
        # if window_mode == 'small':
        #     dn = int(np.round(self.ncols - self.ori_P['z'].shape[1])) + 1
        #     dm = int(np.round(self.nrows - self.ori_P['z'].shape[0])) + 1
        # else:
        #     radius_deg = self.km2deg(self.radius / 1000)
        #     dn = int(np.ceil(radius_deg / self.dlam)) * 2 + 1
        #     dm = int(np.ceil(radius_deg / self.dphi)) * 2 + 1
        #     dn = min(dn, self.ncols)
        #     dm = min(dm, self.nrows)

        i_center = int(np.round((self.LatP[i, j] - self.ori_topo['y'].values[0]) / self.dphi))
        j_center = int(np.round((self.LonP[i, j] - self.ori_topo['x'].values[0]) / self.dlam))
        i_start = max(0, i_center - self.dm // 2)
        i_end = min(self.nrows, i_start + self.dm)
        j_start = max(0, j_center - self.dn // 2)
        j_end = min(self.ncols, j_start + self.dn)
        i_start = i_end - self.dm
        j_start = j_end - self.dn
        return i_start, i_end, j_start, j_end

    def terrain_correction_sequential(self) -> np.ndarray:
        '''
        Compute terrain correction

        Returns
        -------
        tc     : Terrain Correction
        '''
        nrows_P, ncols_P = self.ori_P['z'].shape
        tc = np.zeros((nrows_P, ncols_P))
        # dn = np.round(self.ncols - ncols_P) + 1
        # dm = np.round(self.nrows - nrows_P) + 1
        Hp   = self.ori_P['z'].values 
        
        if self.window_mode == 'small':
            n1 = 0
            n2 = self.dm
            for i in tqdm(range(nrows_P), desc='Computing terrain correction'):
                m1 = 0
                m2 = self.dn
                for j in range(ncols_P):
                    # smallH, smallX, smallY, smallZ = self.get_window(i, j)
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
                    # d[d > self.radius] = np.nan
                    # d[d == 0] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
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
        else:
            # radius-based window
            for i in tqdm(range(nrows_P), desc='Computing terrain correction'):
                # m1 = 0
                # m2 = dm
                for j in range(ncols_P):
                    smallH, smallX, smallY, smallZ = self.get_window(i, j)
                    # Local coordinates (x, y)
                    x = self.coslamp[i, j] * (smallY - self.Yp[i, j]) - \
                        self.sinlamp[i, j] * (smallX - self.Xp[i, j])
                    y = self.cosphip[i, j] * (smallZ - self.Zp[i, j]) - \
                        self.coslamp[i, j] * self.sinphip[i, j] * (smallX - self.Xp[i, j]) - \
                        self.sinlamp[i, j] * self.sinphip[i, j] * (smallY - self.Yp[i, j])
                    # Distances
                    d = np.hypot(x, y)
                    # d = np.where(d <= self.radius, d, np.nan)
                    # d[d > self.radius] = np.nan
                    # d[d == 0] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
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
        return tc

    def terrain_correction_parallel(
        self, 
        chunk_size: int=10, 
        progress: bool=True, 
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
        Hp = self.ori_P['z'].values
        # dn = np.round(self.ncols - ncols_P) + 1
        # dm = np.round(self.nrows - nrows_P) + 1

        

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

        # Precompute window indices as a Numpy array
        window_indices = np.zeros((nrows_P, ncols_P, 4), dtype=np.int32)
        for i in range(nrows_P):
            for j in range(ncols_P):
                window_indices[i, j] = self.get_window_indices(i, j)
        
        # window_indices = [
        #     self.get_window_indices(i, j, window_mode=self.window_mode)
        #     for i in range(nrows_P) for j in range(ncols_P)
        # ]
        results = Parallel(n_jobs=-1)(
            delayed(compute_tc_chunk)(
                row_start, row_end, ncols_P, self.coslamp, self.sinlamp, self.cosphip,
                self.sinphip, Hp, self.ori_topo['z'].values, self.X, self.Y, self.Z, self.Xp,
                self.Yp, self.Zp, self.radius, self.G_rho_dxdy, window_indices
            ) for row_start, row_end in chunks
        )
        # results = Parallel(n_jobs=-1)(
        #     delayed(compute_tc_chunk)(
        #         row_start, row_end, ncols_P, dm, dn, self.coslamp, self.sinlamp, self.cosphip, 
        #         self.sinphip, Hp, self.ori_topo['z'].values, self.X, self.Y, self.Z, self.Xp, 
        #         self.Yp, self.Zp, self.radius, self.G_rho_dxdy
        #     ) for row_start, row_end in chunks
        # )
        
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
            tc = self.terrain_correction_parallel(chunk_size=chunk_size, progress=progress)
        else:
            tc = self.terrain_correction_sequential()
        
        # Save terrain correction
        save_to_netcdf(
            data=tc,
            lon=self.ori_P['x'].values,
            lat=self.ori_P['y'].values,
            dataset_key='tc',
            proj_dir=self.proj_dir,
            overwrite=self.overwrite
        )
        
        return tc

    
    def rtm_anomaly_sequential(self) -> np.ndarray:
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
        dg_RTM   : Residual terrain (RTM) gravity anomalies [mgal]
        
        Reference
        ---------
        1. Forsberg & Tscherning (1984): Topographic effects in gravity field modelling for BVP
           Equation 19
        2. Märdla et al. (2017): From Discrete Gravity Survey Data to a High-resolution Gravity 
           Field Representation in the Nordic-Baltic Region
           Equation 7
        
        Notes
        -----
        1. dg_RTM = 2 * pi * G * rho * (Hp - HrefP) + tc(Href - HrefP) - tc(H - HP)
        
        References
        ---------
        1. Sanso & Sideris: Geoid Determination Theory and Methods, Page 367; Equation 8.73
        '''
        nrows_P, ncols_P = self.ori_P['z'].shape
        tc_rtm = np.zeros((nrows_P, ncols_P))
        Hp     = self.ori_P['z'].values
        Hp_ref = self.ref_P['z'].values
        
        if self.window_mode == 'small':
            n1 = 0
            n2 = self.dn
            for i in tqdm(range(nrows_P), desc='Computing RTM terrain correction'):
                m1 = 0
                m2 = self.dm
                for j in range(ncols_P):
                    smallH     = self.ori_topo['z'].values[n1:n2, m1:m2]
                    smallH_ref = self.ref_topo['z'].values[n1:n2, m1:m2]
                    smallX     = self.X[n1:n2, m1:m2]
                    smallY     = self.Y[n1:n2, m1:m2]
                    smallZ     = self.Z[n1:n2, m1:m2]

                    # Local coordinates (x, y)
                    x = self.coslamp[i, j] * (smallY - self.Yp[i, j]) - \
                        self.sinlamp[i, j] * (smallX - self.Xp[i, j])
                    y = self.cosphip[i, j] * (smallZ - self.Zp[i, j]) - \
                        self.coslamp[i, j] * self.sinphip[i, j] * (smallX - self.Xp[i, j]) - \
                        self.sinlamp[i, j] * self.sinphip[i, j] * (smallY - self.Yp[i, j])

                    # Distances
                    d = np.hypot(x, y)
                    # d[d > self.radius] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
                    d3 = d * d * d
                    d5 = d3 * d * d
                    d7 = d5 * d * d

                    # Height differences
                    DH     = smallH - Hp[i, j]
                    DH_ref = smallH_ref - Hp_ref[i, j]

                    # Powers of height differences
                    DH2 = DH ** 2
                    DH4 = DH2 * DH2
                    DH6 = DH4 * DH2

                    DH_ref2 = DH_ref ** 2
                    DH_ref4 = DH_ref2 * DH_ref2
                    DH_ref6 = DH_ref4 * DH_ref2

                    # Integrate the RTM terrain correction
                    c1 = 0.5 * self.G_rho_dxdy * bn.nansum((DH_ref2 - DH2) / d3)      # 1/2
                    c2 = -0.375 * self.G_rho_dxdy * bn.nansum((DH_ref4 - DH4) / d5)  # 3/8
                    c3 = 0.3125 * self.G_rho_dxdy * bn.nansum((DH_ref6 - DH6) / d7)  # 5/16
                    tc_rtm[i, j] = (c1 + c2 + c3) * 1e5  # [mGal]

                    # Moving window
                    m1 += 1
                    m2 += 1
                n1 += 1
                n2 += 1
        else:
            # radius-based window
            for i in tqdm(range(nrows_P), desc='Computing RTM terrain correction'):
                for j in range(ncols_P):
                    smallH, smallX, smallY, smallZ, smallH_ref = self.get_window(i, j, include_ref=True)
                    # Local coordinates (x, y)
                    x = self.coslamp[i, j] * (smallY - self.Yp[i, j]) - \
                        self.sinlamp[i, j] * (smallX - self.Xp[i, j])
                    y = self.cosphip[i, j] * (smallZ - self.Zp[i, j]) - \
                        self.coslamp[i, j] * self.sinphip[i, j] * (smallX - self.Xp[i, j]) - \
                        self.sinlamp[i, j] * self.sinphip[i, j] * (smallY - self.Yp[i, j])

                    # Distances
                    d = np.hypot(x, y)
                    # d[d > self.radius] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
                    d3 = d * d * d
                    d5 = d3 * d * d
                    d7 = d5 * d * d

                    # Height differences
                    DH     = smallH - Hp[i, j]
                    DH_ref = smallH_ref - Hp_ref[i, j]

                    # Powers of height differences
                    DH2 = DH ** 2
                    DH4 = DH2 * DH2
                    DH6 = DH4 * DH2

                    DH_ref2 = DH_ref ** 2
                    DH_ref4 = DH_ref2 * DH_ref2
                    DH_ref6 = DH_ref4 * DH_ref2

                    # Integrate the RTM terrain correction
                    c1 = 0.5 * self.G_rho_dxdy * bn.nansum((DH_ref2 - DH2) / d3)      # 1/2
                    c2 = -0.375 * self.G_rho_dxdy * bn.nansum((DH_ref4 - DH4) / d5)  # 3/8
                    c3 = 0.3125 * self.G_rho_dxdy * bn.nansum((DH_ref6 - DH6) / d7)  # 5/16
                    tc_rtm[i, j] = (c1 + c2 + c3) * 1e5  # [mGal]
        
        # Calculate RTM gravity anomalies
        dg_RTM = self.two_pi_G_rho * (Hp - Hp_ref) * 1e5 + tc_rtm
        
        return dg_RTM

    
    def rtm_anomaly_parallel(
        self, 
        chunk_size: int=10, 
        progress: bool=True
    ) -> np.ndarray:
        '''
        Compute RTM gravity anomalies (parallelized with chunking).

        Parameters
        ----------
        chunk_size : Chunk size for parallelization
        progress   : Show progress bar

        Returns
        -------
        dg_RTM     : Residual terrain (RTM) gravity anomalies
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
        dg_RTM = np.zeros((nrows_P, ncols_P))
        Hp     = self.ori_P['z'].values
        Hp_ref = self.ref_P['z'].values
        
        # Divide rows into chunks
        chunks = [
            (i, min(i + chunk_size, nrows_P)) 
            for i in range(0, nrows_P, chunk_size)
        ]
        
        print('Computing RTM terrain correction...')
        
        if progress:
            stop_signal = threading.Event()
            progress_thread = threading.Thread(target=print_progress, args=(stop_signal,))
            progress_thread.start()
        
        # Precompute window indices as a Numpy array
        window_indices = np.zeros((nrows_P, ncols_P, 4), dtype=np.int32)
        for i in range(nrows_P):
            for j in range(ncols_P):
                window_indices[i, j] = self.get_window_indices(i, j)
                
        # Submit tasks for each chunk
        results = Parallel(n_jobs=-1)(
            delayed(compute_rtm_tc_chunk)(
                row_start, row_end, ncols_P, self.coslamp, self.sinlamp, self.cosphip,
                self.sinphip, Hp, self.ori_topo['z'].values, self.X, self.Y, self.Z, self.Xp,
                self.Yp, self.Zp, self.radius, self.G_rho_dxdy, Hp_ref, self.ref_topo['z'].values, window_indices
            ) for row_start, row_end in chunks
        )
        
        if progress:
            stop_signal.set()
            progress_thread.join()
            print('\nCompleted.')
        
        # Collect results
        for row_start, row_end, dg_RTM_chunk in results:
            dg_RTM[row_start:row_end, :] = dg_RTM_chunk
        
        # Calculate RTM gravity anomaly
        dg_RTM += self.two_pi_G_rho * (Hp - Hp_ref) * 1e5
        
        return dg_RTM
        
    def rtm_anomaly_approximation(self, tc=None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute RTM gravity anomalies using the approximate formula dg_RTM =  2 * pi * G * rho * (H - Href) - tc
        
        Parameters
        ----------
        tc       : Terrain Correction
        
        Returns
        -------
        dg_RTM   : Residual terrain (RTM) gravity anomalies
        tc       : Terrain Correction
        
        Reference
        ---------
        1. Forsberg & Tscherning (1984): A Study of Terrain Reductions, Density Anomalies and Geophysical Inversion 
                                         Methods in Gravity Field Modelling (Equation 20)
        '''
        if self.ref_topo is None or self.ref_P is None:
            raise ValueError('Reference topography is required for RTM anomaly approximation.')
        if tc is None:
            tc = self.terrain_correction()
        print('Computing RTM gravity anomalies...')
        return (self.two_pi_G_rho * (self.ori_P['z'].values - self.ref_P['z'].values) * 1e5 - tc), tc


    def rtm_anomaly(
        self, 
        parallel: bool=True, 
        chunk_size: int=10,
        approximation: bool=False, 
        tc=None
    ) -> np.ndarray:
        '''
        Parameters
        ----------
        approximation : True/False
                        If True, use the approximate formula dg_RTM =  2 * pi * G * rho * (H - Href) - tc
        parallel      : True/False
                        If True, use the parallelized version. Default: True.

        Returns
        -------
        dg_RTM        : Residual terrain Model (RTM) gravity anomalies
        '''
        if approximation and tc is None:
            dg_RTM = self.rtm_anomaly_approximation()[0]
        elif approximation and tc is not None:
            dg_RTM = self.rtm_anomaly_approximation(tc=tc)[0]
        elif parallel:
            dg_RTM = self.rtm_anomaly_parallel(chunk_size=chunk_size)
        else:
            dg_RTM = self.rtm_anomaly_sequential()
        
        save_to_netcdf(
            data=dg_RTM,
            lon=self.ori_P['x'].values,
            lat=self.ori_P['y'].values,
            dataset_key='rtm',
            proj_dir=self.proj_dir,
            overwrite=self.overwrite
        )
            
        return dg_RTM


    def indirect_effect_sequential(self) -> np.ndarray:
        '''
        Compute the indirect effect due to the second method of Helmert's condensation
        
        Returns
        -------
        ind    : Indirect effect
        
        Notes
        -----
        1. Wichiencharoen (1982): The Indirect Effects On The Computation of Geoid Undulations (Section 2.1.1, Page 21)
        '''
        nrows_P, ncols_P = self.ori_P['z'].shape
        ind  = np.zeros((nrows_P, ncols_P))
        # zeta = np.zeros((nrows_P, ncols_P))
        # Normal gravity at the ellipsoid
        
        gamma_0 = normal_gravity_somigliana(phi=self.LatP, ellipsoid=self.ellipsoid)    
        Hp   = self.ori_P['z'].values 

        if self.window_mode == 'small':
            n1 = 0
            n2 = self.dn
            for i in tqdm(range(nrows_P), desc='Computing terrain correction'):
                m1 = 0
                m2 = self.dm
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
                    # d[d > self.radius] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
                    d3 = d * d * d
                    d5 = d3 * d * d
                    d7 = d5 * d * d

                    # Potential change of regular part
                    dV1 = -np.pi * self.G * self.rho * Hp[i, j] ** 2
                    
                    # Powers of heights
                    Hp3 = Hp[i, j] ** 3
                    Hp5 = Hp3 * Hp[i, j] * Hp[i, j]
                    Hp7 = Hp5 * Hp[i, j] * Hp[i, j]
                    H3  = smallH ** 3
                    H5  = H3 * smallH * smallH
                    H7  = H5 * smallH * smallH
                    
                    v2  = -1/6 * bn.nansum((H3 - Hp3) / d3)
                    v3  = 0.075 * bn.nansum((H5 - Hp5) / d5)     # 3/40
                    v4  = -15/336 * bn.nansum((H7 - Hp7) / d7)   
                    dV2 = self.G_rho_dxdy * (v2 + v3 + v4)
                    
                    # Total potential change
                    dV = dV1 + dV2

                    # Indirect effect
                    ind[i, j] = dV / gamma_0[i, j]
                    # zeta[i, j] = 1 / 9.82 * dV2
                    
                    # moving window
                    m1 += 1
                    m2 += 1
                n1 += 1
                n2 += 1
        else:
            # radius-based window
            for i in tqdm(range(nrows_P), desc='Computing terrain correction'):
                for j in range(ncols_P):
                    smallH, smallX, smallY, smallZ = self.get_window(i, j)
                    # Local coordinates (x, y)
                    x = self.coslamp[i, j] * (smallY - self.Yp[i, j]) - \
                        self.sinlamp[i, j] * (smallX - self.Xp[i, j])
                    y = self.cosphip[i, j] * (smallZ - self.Zp[i, j]) - \
                        self.coslamp[i, j] * self.sinphip[i, j] * (smallX - self.Xp[i, j]) - \
                        self.sinlamp[i, j] * self.sinphip[i, j] * (smallY - self.Yp[i, j])

                    # Distances
                    d = np.hypot(x, y)
                    # d[d > self.radius] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
                    d3 = d * d * d
                    d5 = d3 * d * d
                    d7 = d5 * d * d

                    # Potential change of regular part
                    dV1 = -np.pi * self.G * self.rho * Hp[i, j] ** 2
                    
                    # Powers of heights
                    Hp3 = Hp[i, j] ** 3
                    Hp5 = Hp3 * Hp[i, j] * Hp[i, j]
                    Hp7 = Hp5 * Hp[i, j] * Hp[i, j]
                    H3  = smallH ** 3
                    H5  = H3 * smallH * smallH
                    H7  = H5 * smallH * smallH
                    
                    v2  = -1/6 * bn.nansum((H3 - Hp3) / d3)
                    v3  = 0.075 * bn.nansum((H5 - Hp5) / d5)
                    v4  = -15/336 * bn.nansum((H7 - Hp7) / d7)
                    dV2 = self.G_rho_dxdy * (v2 + v3 + v4)
                    
                    # Total potential change
                    dV = dV1 + dV2

                    # Indirect effect
                    ind[i, j] = dV / gamma_0[i, j]
                    # zeta[i, j] = 1 / 9.82 * dV2
                    
        return ind
    
    def indirect_effect_parallel(
        self, 
        chunk_size: int=10, 
        progress: bool=True, 
    ) -> np.ndarray:
        '''
        Compute terrain correction (parallelized with chunking).

        Parameters
        ----------
        chunk_size : number of rows to process in each chunk
        progress   : Progress bar display

        Returns
        -------
        ind        : Indirect effect
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
        dV2 = np.zeros((nrows_P, ncols_P)) # Potential change of irregular part
        Hp = self.ori_P['z'].values

        # Potential change of the regular part of topography
        dV1 = -np.pi * self.G * self.rho * Hp ** 2
        # Normal gravity at the ellipsoid
        gamma_0 = normal_gravity_somigliana(phi=self.LatP, ellipsoid=self.ellipsoid)
        
        # Divide rows into chunks
        chunks = [
            (i, min(i + chunk_size, nrows_P)) 
            for i in range(0, nrows_P, chunk_size)
        ]

        print('Computing potential change of irregular part...') 
        
        if progress:
            stop_signal = threading.Event()
            progress_thread = threading.Thread(target=print_progress, args=(stop_signal,))
            progress_thread.start()

        # Precompute window indices as a NumPy array
        window_indices = np.zeros((nrows_P, ncols_P, 4), dtype=np.int32)
        for i in range(nrows_P):
            for j in range(ncols_P):
                window_indices[i, j] = self.get_window_indices(i, j)
        
        # Submit tasks for each chunk
        results = Parallel(n_jobs=-1)(
            delayed(compute_ind_chunk)(
                row_start, row_end, ncols_P, self.coslamp, self.sinlamp, self.cosphip,
                self.sinphip, Hp, self.ori_topo['z'].values, self.X, self.Y, self.Z, self.Xp,
                self.Yp, self.Zp, self.radius, self.G_rho_dxdy, Hp, self.ori_topo['z'].values, window_indices
            ) for row_start, row_end in chunks
        )
        
        if progress:
            stop_signal.set()
            progress_thread.join()
            print('\nCompleted.')
        
        # Collect results
        for row_start, row_end, ind_chunk in results:
            dV2[row_start:row_end, :] = ind_chunk
        
        # Total potential change
        dV = dV1 + dV2
        # Compute indirect effect
        ind = dV / gamma_0
           
        return ind

    def indirect_effect(
        self,
        parallel: bool=True,
        chunk_size: int=10,
        progress: bool=True,
    ) -> np.ndarray:
        '''
        Compute indirect effect.

        Parameters
        ----------
        parallel   : If True, use the parallelized version. Default: True.
        chunk_size : Size of the chunk in terms of number of rows. Default is 10.
        progress   : If True, display a progress bar. Default: True.

        Return
        ------
        ind        : Indirect effect
        '''
        if parallel:
            ind = self.indirect_effect_parallel(chunk_size=chunk_size, progress=progress)
        else:
            ind = self.indirect_effect_sequential()
        
        # Save indirect effect
        save_to_netcdf(
            data=ind,
            lon=self.ori_P['x'].values,
            lat=self.ori_P['y'].values,
            dataset_key='IND',
            proj_dir=self.proj_dir,
            overwrite=self.overwrite
        )

        return ind


    def rtm_height_anomaly_sequential(self) -> np.ndarray:
        '''
        Compute RTM height anomaly (sequential).

        Returns
        -------
        z_rtm : RTM height anomaly [m]

        Reference
        ---------
        1. Forsberg & Tscherning (1984): Topographic effects in gravity field modelling for BVP
        '''
        if self.ref_topo is None or self.ref_P is None:
            raise ValueError("Reference topography (ref_topo) is required for RTM height anomaly computation")

        nrows_P, ncols_P = self.ori_P['z'].shape
        z_rtm = np.zeros((nrows_P, ncols_P))
        Hp = self.ori_P['z'].values
        HrefP = self.ref_P['z'].values

        if self.window_mode == 'small':
            n1 = 0
            n2 = self.dn
            for i in tqdm(range(nrows_P), desc='Computing RTM height anomaly'):
                m1 = 0
                m2 = self.dm
                for j in range(ncols_P):
                    smallH = self.ori_topo['z'].values[n1:n2, m1:m2]
                    smallH_ref = self.ref_topo['z'].values[n1:n2, m1:m2]
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
                    # d[d > self.radius] = np.nan
                    # d[d == 0] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
                    d3 = d * d * d
                    d5 = d3 * d * d

                    # Height differences
                    z1 = smallH - smallH_ref
                    z3 = smallH**3 - smallH_ref**3
                    z5 = smallH**5 - smallH_ref**5

                    # Integrate the RTM height anomaly
                    c1 = bn.nansum(z1 / d)
                    c2 = -1/6 * bn.nansum(z3 / d3)
                    c3 = 0.075 * bn.nansum(z5 / d5)  # 3/40
                    z_rtm[i, j] = (1 / 9.82) * self.G_rho_dxdy * (c1 + c2 + c3)

                    # Moving window
                    m1 += 1
                    m2 += 1
                n1 += 1
                n2 += 1
        else:
            for i in tqdm(range(nrows_P), desc='Computing RTM height anomaly'):
                for j in range(ncols_P):
                    smallH, smallX, smallY, smallZ, smallH_ref = self.get_window(i, j, include_ref=True)
                    x = self.coslamp[i, j] * (smallY - self.Yp[i, j]) - \
                        self.sinlamp[i, j] * (smallX - self.Xp[i, j])
                    y = self.cosphip[i, j] * (smallZ - self.Zp[i, j]) - \
                        self.coslamp[i, j] * self.sinphip[i, j] * (smallX - self.Xp[i, j]) - \
                        self.sinlamp[i, j] * self.sinphip[i, j] * (smallY - self.Yp[i, j])
                    d = np.hypot(x, y)
                    # d[d > self.radius] = np.nan
                    # d[d == 0] = np.nan
                    d[(d > self.radius) | (d == 0)] = np.nan
                    d3 = d * d * d
                    d5 = d3 * d * d
                    z1 = smallH - smallH_ref
                    z3 = smallH**3 - smallH_ref**3
                    z5 = smallH**5 - smallH_ref**5
                    c1 = bn.nansum(z1 / d)
                    c2 = -1/6 * bn.nansum(z3 / d3)
                    c3 = 0.075 * bn.nansum(z5 / d5)
                    z_rtm[i, j] = (1 / 9.82) * self.G_rho_dxdy * (c1 + c2 + c3)

        return z_rtm

    def rtm_height_anomaly_parallel(self, chunk_size: int=10, progress: bool=True) -> np.ndarray:
        '''
        Compute RTM height anomaly (parallelized with chunking).

        Parameters
        ----------
        chunk_size : Number of rows to process in each chunk
        progress   : If True, display a progress indicator

        Returns
        -------
        z_rtm : RTM height anomaly [m]
        '''
        if self.ref_topo is None or self.ref_P is None:
            raise ValueError("Reference topography (ref_topo) is required for RTM height anomaly computation")

        if progress:
            def print_progress(stop_signal) -> None:
                while not stop_signal.is_set():
                    sys.stdout.write("#")
                    sys.stdout.flush()
                    time.sleep(1.5)

        nrows_P, ncols_P = self.ori_P['z'].shape
        z_rtm = np.zeros((nrows_P, ncols_P))
        Hp = self.ori_P['z'].values
        HrefP = self.ref_P['z'].values

        chunks = [(i, min(i + chunk_size, nrows_P)) for i in range(0, nrows_P, chunk_size)]
        print('Computing RTM height anomaly...')

        if progress:
            stop_signal = threading.Event()
            progress_thread = threading.Thread(target=print_progress, args=(stop_signal,))
            progress_thread.start()

        window_indices = np.zeros((nrows_P, ncols_P, 4), dtype=np.int32)
        for i in range(nrows_P):
            for j in range(ncols_P):
                window_indices[i, j] = self.get_window_indices(i, j)

        results = Parallel(n_jobs=-1)(
            delayed(compute_rtm_height_anomaly_chunk)(
                row_start, row_end, ncols_P, self.coslamp, self.sinlamp, self.cosphip,
                self.sinphip, Hp, self.ori_topo['z'].values, self.X, self.Y, self.Z, self.Xp,
                self.Yp, self.Zp, self.radius, self.G_rho_dxdy, HrefP, self.ref_topo['z'].values, window_indices
            ) for row_start, row_end in chunks
        )

        if progress:
            stop_signal.set()
            progress_thread.join()
            print('\nCompleted.')

        for row_start, row_end, z_rtm_chunk in results:
            z_rtm[row_start:row_end, :] = z_rtm_chunk

        return z_rtm

    def rtm_height_anomaly(self, parallel: bool=True, chunk_size: int=10, progress: bool=True) -> np.ndarray:
        '''
        Compute RTM height anomaly.

        Parameters
        ----------
        parallel   : If True, use the parallelized version. Default: True.
        chunk_size : Size of the chunk in terms of number of rows. Default is 10.
        progress   : If True, display a progress indicator. Default: True.

        Returns
        -------
        z_rtm : RTM height anomaly [m]
        '''
        if parallel:
            z_rtm = self.rtm_height_anomaly_parallel(chunk_size=chunk_size, progress=progress)
        else:
            z_rtm = self.rtm_height_anomaly_sequential()

        save_to_netcdf(
            data=z_rtm,
            lon=self.ori_P['x'].values,
            lat=self.ori_P['y'].values,
            dataset_key='zeta',
            proj_dir=self.proj_dir,
            overwrite=self.overwrite
        )

        return z_rtm
    
    
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
