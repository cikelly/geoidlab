############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import lzma
# import os

import numpy as np
import pandas as pd

from numba import jit
from numba_progress import ProgressBar
from tqdm import tqdm
from pathlib import Path

from easy_pygeoid.legendre import ALFsGravityAnomaly, ALF
import easy_pygeoid.coordinates as co

import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar as DaskProgressBar

class DigitalTerrainModel:
    def __init__(self, model_name=None, nmax=2190, ellipsoid='wgs84') -> None:
        '''
        Initialize the DigitalTerrainModel class
        
        Parameters
        ----------
        model_name : Name of the DTM model file
        nmax       : Maximum degree of spherical harmonics
        ellipsoid  : Reference ellipsoid

        Returns
        -------
        None
        '''
        self.name = model_name
        self.nmax = nmax
        self.ellipsoid = ellipsoid
        # self.progress = show_progress

        if self.name is None:
            # script_dir = os.path.dirname(__file__)
            script_dir: Path = Path(__file__).resolve().parent
            # self.name = os.path.join(script_dir, 'data', 'DTM2006.xz')
            self.name = script_dir / 'data' / 'DTM2006.xz'
            print(f'Using compressed file in src/data directory ...')
            with lzma.open(self.name, 'rt') as f:
                self.dtm = f.readlines()
        else:
            with open(self.name, 'r') as f:
                self.dtm = f.readlines() # self.dtm is the DTM2006 text file

    def read_dtm2006(self) -> None:
        '''
        Read DTM data stored as compressed LZMA file or the original DTM2006 file
        '''
        HCnm = np.zeros((self.nmax + 1, self.nmax + 1))
        HSnm = np.zeros((self.nmax + 1, self.nmax + 1))

        for line in self.dtm:
            line = line.split()

            n = int(line[0])
            m = int(line[1])

            if n > self.nmax:
                break

            if n <= self.nmax + 1 and m <= self.nmax + 1:
                HCnm[n, m] = float(line[2].replace('D', 'E'))
                HSnm[n, m] = float(line[3].replace('D', 'E'))
        self.HCnm = HCnm
        self.HSnm = HSnm

    @staticmethod
    @jit(nopython=True)
    def compute_height_chunk(HCnm, HSnm, lon, n, Pnm) -> np.ndarray:
        '''
        Compute a chunk of heights for a specific degree n using Numba for optimization

        Parameters
        ----------
        HCnm, HSnm : Spherical Harmonic Coefficients (2D arrays)
        lon        : Longitude (1D arrays)
        n          : Specific degree
        Pnm        : Associated Legendre functions (3D array)

        Returns
        -------
        H          : Heights (1D array)
        '''
        H = np.zeros(len(lon))
        for m in range(n + 1):
            H += (HCnm[n, m] * np.cos(m * lon) + HSnm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
        return H
    
    def calculate_height_chunk(self, lon, lat, Pnm, progress=True) -> np.ndarray:
        '''
        Calculate height for a chunk of data

        Parameters
        ----------
        lon        : geodetic/geographic longitude of point(s) of interest (degrees)
        lat        : geodetic/geographic latitude of point(s) of interest  (degrees)
        Pnm        : fully normalized associated Legendre functions
        chunk_size : size of the chunk of data to process at once
        ellipsoid  : reference ellipsoid

        Returns
        -------
        H_chunk    : Synthesized height for the chunk
        '''
        lon = np.radians(lon)
        lat = np.radians(lat)
        HCnm = self.HCnm
        HSnm = self.HSnm

        H_chunk = np.zeros(len(lon))

        if progress:
            # with ProgressBar(
            #         total=self.nmax + 1, leave=bool(chunk_num == n_chunk - 1),
            #         desc=f'Synthesizing heights from DTM: {chunk_num+1}/{n_chunk}'
            #     ) as pbar:
            for n in range(self.nmax + 1):
                H_chunk += DigitalTerrainModel.compute_height_chunk(HCnm, HSnm, lon, n, Pnm)
                # pbar.update(1)
        else:
            # print('Computing heights...')
            for n in range(self.nmax + 1):
                H_chunk += DigitalTerrainModel.compute_height_chunk(HCnm, HSnm, lon, n, Pnm)
                
        return H_chunk
    
    def calculate_height_dask(
        self, 
        lon, 
        lat, 
        Pnm=None, 
        chunk_size=100, 
        split_data=False, 
        progress=True, 
        leg_progress=False
    ) -> np.ndarray:
        # Convert to numpy arrays if they're Pandas Series
        if isinstance(lon, pd.Series):
            lon = lon.values
        if isinstance(lat, pd.Series):
            lat = lat.values
        # Convert lon, lat to radians and initialize outputs
        # lon = np.radians(lon)
        # lat = np.radians(lat)
        
        # HCnm = self.HCnm
        # HSnm = self.HSnm
        
        # if Pnm is None:
        #     Pnm = ALFsGravityAnomaly(phi=lat, lambd=lon, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=leg_progress)

        def compute_chunk(start_idx, end_idx) -> np.ndarray:
            lon_chunk = lon[start_idx:end_idx]
            lat_chunk = lat[start_idx:end_idx]
            Pnm_chunk = ALFsGravityAnomaly(phi=lat_chunk, lambd=lon_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=leg_progress)
            return self.calculate_height_chunk(lon_chunk, lat_chunk, Pnm_chunk, progress=False)
        
        n_points = len(lon)
        indices = [(i * chunk_size, min((i + 1) * chunk_size, n_points)) for i in range((n_points // chunk_size) + 1)]
        
        # Use a delayed computation for each chunk and save the intermediate results to disk
        results = [delayed(compute_chunk)(start_idx, end_idx) for start_idx, end_idx in indices]

        # Compute the results using Dask's disk cache to limit memory usage
        with DaskProgressBar():
            results = compute(*results, scheduler='threads')

        # Concatenate results into the final array
        return np.concatenate(results)

    def calculate_height(
        self, 
        lon, 
        lat, 
        Pnm=None, 
        chunk_size=100, 
        split_data=False, 
        progress=True, 
        leg_progress=False
    ) -> np.ndarray:
        '''
        Wrapper function to handle data and call the Numba-optimized function

        Parameters
        ----------
        lon        : geodetic/geographic longitude of point(s) of interest (degrees)
        lat        : geodetic/geographic latitude of point(s) of interest  (degrees)
        nmax       : maximum spherical harmonic degree of expansion
        Pnm        : fully normalized associated Legendre functions
        chunk_size : size of the chunk of data to process at once
        ellipsoid  : reference ellipsoid
        split_data : whether to split the data into chunks

        Returns
        -------
        Hdtm       : Synthesized height

        References
        ----------
        1. Hirt et al.(2010): Combining EGM2008 and SRTM/DTM2006.0 residual terrain model 
           data to improve quasigeoid computations in mountainous areas devoid of gravity data
           Eq. 4
        2. Pavlis et al. (2007): Terrain-related gravimetric quantities computed for the next EGM
           Eq. 1
        '''
        if isinstance(lon, pd.Series):
            lon = lon.values
        if isinstance(lat, pd.Series):
            lat = lat.values

        if not split_data:
            if Pnm is None:
                Pnm = ALFsGravityAnomaly(phi=lat, lambd=lon, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=leg_progress)
            
            lon = np.radians(lon)
            lat = np.radians(lat)
            HCnm = self.HCnm
            HSnm = self.HSnm

            H = np.zeros(len(lon))

            if progress:
                with ProgressBar(total=self.nmax + 1, desc='Synthesizing heights from DTM') as pbar:
                    for n in range(self.nmax + 1):
                        H += DigitalTerrainModel.compute_height_chunk(HCnm, HSnm, lon, n, Pnm)
                        pbar.update(1)
            else:
                # print('Computing heights...')
                for n in range(self.nmax + 1):
                    H += DigitalTerrainModel.compute_height_chunk(HCnm, HSnm, lon, n, Pnm)
        else:
            n_points = len(lon)
            n_chunks = (n_points // chunk_size) + 1
            # print(f'Data will be processed in {n_chunks} chunks...\n')
            
            H = np.zeros(n_points)
            
            # with tqdm(total=n_chunks, desc='Processing chunks') as pbar:
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_points)

                lon_chunk = lon[start_idx:end_idx]
                lat_chunk = lat[start_idx:end_idx]

                Pnm_chunk = ALFsGravityAnomaly(phi=lat_chunk, lambd=lon_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=leg_progress)

                H[start_idx:end_idx] = self.calculate_height_chunk(lon_chunk, lat_chunk, Pnm_chunk, progress=progress)
                # pbar.update(1)

                Pnm_chunk = None
        return H
    
    def calculate_height_2D(self, lon=None, lat=None, grid_spacing=1, progress=True, chunk_size=65341) -> np.ndarray:
        '''
        Vectorized computations of height on a grid
        
        Parameters
        ----------
        lon         : Longitude (1D array) -- units of degree
        lat         : Latitude (1D array)  -- units of degree
        grid_spacing: grid spacing (in degrees)
        chunk_size  : Maximum size of each chunk to process

        Returns
        -------
        H      : Height array (2D array)
        '''
        self.lon = lon
        self.lat = lat
        self.grid_spacing = grid_spacing
        
        if self.lon is None and self.lat is None:
            print('No grid coordinates provided. Computing over the entire globe...\n')
            self.lambda_ = np.radians(np.arange(-180, 180+self.grid_spacing, self.grid_spacing))
            self.theta   = np.radians(np.arange(0, 180+self.grid_spacing, self.grid_spacing)) # co-latitude
            self.lon     = np.arange(-180, 180+self.grid_spacing, self.grid_spacing)
            self.lat     = np.arange(-90, 90+self.grid_spacing, self.grid_spacing)
        else:
            self.lon = lon.flatten()
            self.lat = lat.flatten()
            self.r, self.theta, _ = co.geodetic2spherical(
                phi=self.lat, lambd=self.lon, 
                height=np.zeros(self.lon.shape), ellipsoid=self.ellipsoid
            )
            self.lambda_ = np.radians(self.lon)
        
        if self.lon.ndim == 1:
            self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
            self.Lat *= -1
        
        # Precompute for vectorization
        self.n = np.arange(self.nmax + 1)
        self.cosm = np.cos(np.arange(0, self.nmax+1)[:, np.newaxis] * self.lambda_)
        self.sinm = np.sin(np.arange(0, self.nmax+1)[:, np.newaxis] * self.lambda_)        
        
        degree_term = np.ones(len(self.n))
        # Set degrees 0 and 1 to zero
        
        H = np.zeros((len(self.theta), len(self.lambda_)))
        
        # Split the grid into chunks if necessary
        num_chunks = int(np.ceil(len(self.lon) / chunk_size))
        lon_chunks = np.array_split(self.lon, num_chunks)
        lat_chunks = np.array_split(self.lat, num_chunks)
        
        for lon_chunk, lat_chunk in zip(lon_chunks, lat_chunks):
            lambda_chunk = np.radians(lon_chunk)
            theta_chunk = np.radians(lat_chunk)
            
            cosm_chunk = np.cos(np.arange(0, self.nmax+1)[:, np.newaxis] * lambda_chunk)
            sinm_chunk = np.sin(np.arange(0, self.nmax+1)[:, np.newaxis] * lambda_chunk)
            
            H_chunk = np.zeros((len(theta_chunk), len(lambda_chunk)))
            
            if progress:
                for i, theta_ in tqdm(enumerate(theta_chunk), total=len(theta_chunk), desc='Computing heights'):
                    Pnm = ALF(vartheta=theta_, nmax=self.nmax, ellipsoid=self.ellipsoid)
                    H_chunk[i,:] = degree_term @ ((self.HCnm * Pnm) @ cosm_chunk + (self.HSnm * Pnm) @ sinm_chunk)
            else:
                print('Computing heights...')
                for i, theta_ in enumerate(theta_chunk):
                    Pnm = ALF(vartheta=theta_, nmax=self.nmax, ellipsoid=self.ellipsoid)
                    H_chunk[i,:] = degree_term @ ((self.HCnm * Pnm) @ cosm_chunk + (self.HSnm * Pnm) @ sinm_chunk)
            
            # Merge the chunk results into the final height array
            H[:len(theta_chunk), :len(lambda_chunk)] = H_chunk
        
        return self.Lon, self.Lat, H

# TODO: Implement function for chunking data into manageble square/rectangular sizes that can then be passed to workers
def auto_chunk(self, chunk_size: int = 1000) -> list:
    """
    Automatically chunk data into manageable square/rectangular sizes that can then be passed to workers
    
    Parameters
    ----------
    chunk_size : int
        Size of the chunk in terms of number of lat/lon points. Default is 1000.
    """
    data_size = len(self.lon)
    n_chunks = int(np.ceil(data_size / chunk_size))
    chunk_indices = np.array_split(np.arange(data_size), n_chunks)
    chunks = []
    for start, end in chunk_indices:
        lon_chunk = self.lon[start:end]
        lat_chunk = self.lat[start:end]
        chunks.append((lon_chunk, lat_chunk))
    return chunks
