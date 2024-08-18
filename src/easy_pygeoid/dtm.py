############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import lzma
import os
import numpy as np
import pandas as pd
from numba import jit
from numba_progress import ProgressBar
from legendre import ALFsGravityAnomaly, ALF
import coordinates as co
from tqdm import tqdm

class DigitalTerrainModel:
    def __init__(self, model_name=None, nmax=2190, ellipsoid='wgs84'):
        self.name = model_name
        self.nmax = nmax
        self.ellipsoid = ellipsoid

        if self.name is None:
            script_dir = os.path.dirname(__file__)
            self.name = os.path.join(script_dir, 'data', 'DTM2006.xz')
            print(f'Using compressed file in src/data directory ...')
            with lzma.open(self.name, 'rt') as f:
                self.dtm = f.readlines()
        else:
            with open(self.name, 'r') as f:
                self.dtm = f.readlines()

    def read_dtm2006(self):
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
    def compute_height_chunk(HCnm, HSnm, lon, n, Pnm):
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
    
    def calculate_height_chunk(self, lon, lat, Pnm):
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

        with ProgressBar(total=self.nmax + 1, desc='Synthesizing heights from DTM') as pbar:
            for n in range(self.nmax + 1):
                H_chunk += self.compute_height_chunk(HCnm, HSnm, lon, n, Pnm)
                pbar.update(1)

        return H_chunk

    def calculate_height(self, lon, lat, Pnm=None, chunk_size=100, split_data=False):
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
        1. Hirt et al.(2010):Combining EGM2008 and SRTM/DTM2006.0 residual terrain model 
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
                Pnm = ALFsGravityAnomaly(phi=lat, lambd=lon, nmax=self.nmax, ellipsoid=self.ellipsoid)
            
            lon = np.radians(lon)
            lat = np.radians(lat)
            HCnm = self.HCnm
            HSnm = self.HSnm

            H = np.zeros(len(lon))

            with ProgressBar(total=self.nmax + 1, desc='Synthesizing heights from DTM') as pbar:
                for n in range(self.nmax + 1):
                    H += self.compute_height_chunk(HCnm, HSnm, lon, n, Pnm)
                    pbar.update(1)
        else:
            n_points = len(lon)
            n_chunks = (n_points // chunk_size) + 1
            print(f'Data will be processed in {n_chunks} chunks...\n')
            
            H = np.zeros(n_points)

            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_points)

                lon_chunk = lon[start_idx:end_idx]
                lat_chunk = lat[start_idx:end_idx]

                print(f'Processing chunk {i + 1} of {n_chunks}...')
                Pnm_chunk = ALFsGravityAnomaly(phi=lat_chunk, lambd=lon_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid)

                H[start_idx:end_idx] = self.calculate_height_chunk(lon_chunk, lat_chunk, Pnm_chunk)
                print('\n')

                Pnm_chunk = None
        return H
    
    def calculate_height_2D(self, lon=None, lat=None, grid_spacing=1):
        '''
        Vectorized computations of height on a grid
        
        Parameters
        ----------
        lon         : Longitude (1D array) -- units of degree
        lat         : Latitude (1D array)  -- units of degree
        grid_spacing: grid spacing (in degrees)

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
            self.r, self.theta, _ = co.geodetic2spherical(
                phi=self.lat, lambd=self.lon, 
                height=len(self.lon), ellipsoid=self.ellipsoid
            )
            self.lambda_ = np.radians(self.lon)
        
        
        self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
        self.Lat *= -1
        
        # Precompute for vectorization
        self.n = np.arange(self.nmax + 1)
        self.cosm = np.cos(np.arange(0, self.nmax+1)[:, np.newaxis] * self.lambda_)
        self.sinm = np.sin(np.arange(0, self.nmax+1)[:, np.newaxis] * self.lambda_)        
        
        degree_term = np.ones(len(self.n))
        # Set degrees 0 and 1 to zero
        
        H = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing heights'):
            Pnm = ALF(vartheta=theta_, nmax=self.nmax, ellipsoid=self.ellipsoid)
            H[i,:] = degree_term @ ((self.HCnm * Pnm) @ self.cosm + (self.HSnm * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, H
