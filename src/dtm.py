############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import lzma
import os
import constants

import numpy as np
import pandas as pd

from legendre import ALFsGravityAnomaly
from numba import jit
from numba_progress import ProgressBar
# from tqdm import tqdm

class DigitalTerrainModel:
    def __init__(self, model_name=None, nmax=2190):
        self.name = model_name
        self.nmax = nmax
        
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

        HCnm = np.zeros((self.nmax+1, self.nmax+1))
        HSnm = np.zeros((self.nmax+1, self.nmax+1))
            
            
        for line in self.dtm:
            line = line.split()
            
            n = int(line[0])
            m = int(line[1])

            if n > self.nmax:
                break
            
            if n <= self.nmax+1 and m <= self.nmax+1:
                HCnm[n,m] = float(line[2].replace('D', 'E'))
                HSnm[n,m] = float(line[3].replace('D', 'E'))
            self.HCnm = HCnm
            self.HSnm = HSnm
        # return self.HCnm, self.HSnm

    @jit(nopython=True)
    def compute_height_chunk(HCnm, HSnm, lon, lat, n, Pnm):
        '''
        Compute a chunk of heights for a specific degree n using Numba for optimization

        Parameters
        ----------
        HCnm, HSnm : Spherical Harmonic Coefficients (2D arrays)
        lon        : Longitude (1D arrays)
        lat        : Latitude (1D arrays)
        n          : Specific degree
        Pnm        : Associated Legendre functions (3D array)

        Returns
        -------
        H          : Heights (1D array)
        '''
        H = np.zeros(len(lon))
        for m in range(n+1):
            H += (HCnm[n, m] * np.cos(m * lon) + HSnm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
        # H = sum  
        
        return H  
            
    def calculate_height(self, lon, lat, Pnm=None, ellipsoid='wgs84'):
        '''
        Wrapper function to handle data and call the Numba-optimized function
        
        Parameters
        ----------
        lon        : geodetic/geographic longitude of point(s) of interest (degrees)
        lat        : geodetic/geographic latitude of point(s) of interest  (degrees)
        nmax       : maximum spherical harmonic degree of expansion
        Pnm        : fully normalized associated Legendre functions

        Returns
        -------
        Hdtm       : Synthesized height

        References
        ----------
        1. Sanso and Sideris (2013): Geoid Determination Theory and Methods, Chapter 8. (Eq. 8.79)
        '''
        if Pnm is None:
            Pnm = ALFsGravityAnomaly(phi=lat, lambd=lon, nmax=self.nmax)
        
        if isinstance(lon, pd.Series):
            lon = lon.values
        if isinstance(lat, pd.Series):
            lat = lat.values
        
        lon = np.radians(lon)
        lat = np.radians(lat)
        
        H = np.zeros(lon)
        
        with ProgressBar(total=self.nmax-1, desc='Synthesizing heights from DTM') as pbar:
            for n in range(2, self.nmax+1):
                H += self.compute_height_chunk(self.HCnm, self.HSnm, lon, lat, n, Pnm)
                pbar.update(1)
        
        return H