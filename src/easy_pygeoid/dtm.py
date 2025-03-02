############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import lzma

from pathlib import Path
from . import coordinates as co
from easy_pygeoid.legendre import ALF, ALFsGravityAnomaly
from .utils.parallel_utils import compute_harmonic_sum
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import numpy as np


def process_chunk(args) -> np.ndarray:
    self, lon_chunk, lat_chunk, height_chunk, nmax, ellipsoid, leg_progress, chunk_idx = args
    _, theta, _ = co.geodetic2spherical(phi=lat_chunk, lambd=lon_chunk, ellipsoid=ellipsoid, height=height_chunk)
    _lambda = np.radians(lon_chunk)
    m = np.arange(nmax + 1)
    mlambda = m[:, np.newaxis] * _lambda
    cosm = np.cos(mlambda)
    sinm = np.sin(mlambda)
    
    Pnm = ALFsGravityAnomaly(vartheta=theta, nmax=nmax, ellipsoid=ellipsoid, show_progress=leg_progress)
    
    # H = 

    return compute_harmonic_sum(Pnm, self.HCnm, self.HSnm, cosm, sinm)

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
            script_dir: Path = Path(__file__).resolve().parent
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
    
    def dtm2006_height_point(
        self,
        lon: float,
        lat: float,
        height=None
    ) -> float:    
        '''
        Compute height for a single point using DTM2006.0 spherical harmonics
        
        Parameters
        ----------
        lon       : geodetic longitude
        lat       : geodetic latitude
        
        Returns
        -------
        H         : synthesized height
        '''
        # Check if self has the HCnm attribute
        if not hasattr(self, 'HCnm') or not hasattr(self, 'HSnm'):
            self.read_dtm2006()
            
        if height is None:
            height = 0
        
        _, theta, _ = co.geodetic2spherical(
            phi=np.array([lat]), 
            lambd=np.array([lon]), 
            ellipsoid=self.ellipsoid,
            height=height
        )
        
        theta = theta[0]
        _lambda = np.radians(lon)
        m = np.arange(self.nmax + 1)[:, np.newaxis]
        mlambda = m * _lambda
        cosm = np.cos(mlambda)
        sinm = np.sin(mlambda)
        
        Pnm = ALF(vartheta=theta, nmax=self.nmax, ellipsoid=self.ellipsoid)
        H = np.sum((self.HCnm * Pnm) @ cosm + (self.HSnm * Pnm) @ sinm)
        return float(H)
    
    # def dtm2006_height(
    #     self,
    #     lon: np.ndarray,
    #     lat: np.ndarray,
    #     chunk_size: int = 100,
    #     leg_progress: bool = False,
    #     height: np.ndarray = None
    # ) -> np.ndarray:
    #     '''
    #     Compute heights from DTM2006.0 spherical harmonic coefficients
        
    #     Parameters
    #     ----------
    #     lon         : geodetic longitude
    #     lat         : geodetic latitude
    #     chunk_size  : number of points to process at a time
    #     leg_progress: show progress bar for Legendre polynomial computation
    #     height      : height above ellipsoid (optional)
        
    #     Returns
    #     -------
    #     H           : synthesized height
    #     '''
    #     import numpy as np
    #     from tqdm import tqdm
    #     import easy_pygeoid.coordinates as co
    #     from easy_pygeoid.legendre import ALFsGravityAnomaly

    #     lon = np.asarray(lon)
    #     lat = np.asarray(lat)
        
    #     if lon.shape != lat.shape:
    #         raise ValueError('lon and lat must have the same shape')

    #     input_shape = lon.shape
    #     lon_flat = lon.flatten()
    #     lat_flat = lat.flatten()
    #     height_flat = np.zeros_like(lon_flat) if height is None else height.flatten()
    #     num_points = len(lon_flat)
        
    #     if num_points == 1:
    #         return self.dtm2006_height_point(lon_flat[0], lat_flat[0])

    #     H_flat = np.zeros(num_points)
        
    #     if num_points <= chunk_size:
    #         r, theta, _ = co.geodetic2spherical(
    #             phi=lat_flat, 
    #             lambd=lon_flat, 
    #             ellipsoid=self.ellipsoid,
    #             height=height_flat
    #         )
    #         _lambda = np.radians(lon_flat)
    #         m = np.arange(self.nmax + 1)
    #         mlambda = m[:, np.newaxis] * _lambda
    #         cosm = np.cos(mlambda)  # (nmax+1, num_points)
    #         sinm = np.sin(mlambda)  # (nmax+1, num_points)
            
    #         Pnm = ALFsGravityAnomaly(
    #             vartheta=theta, 
    #             nmax=self.nmax, 
    #             ellipsoid=self.ellipsoid,
    #             show_progress=leg_progress
    #         )
            
    #         H_flat = (np.einsum('inm,nm,im->i', Pnm, self.HCnm, cosm.T) + 
    #                 np.einsum('inm,nm,im->i', Pnm, self.HSnm, sinm.T))
    #     else:
    #         for start in tqdm(range(0, num_points, chunk_size), desc='Computing chunks'):
    #             end = min(start + chunk_size, num_points)
    #             lon_chunk = lon_flat[start:end]
    #             lat_chunk = lat_flat[start:end]
    #             height_chunk = height_flat[start:end]
                
    #             r, theta, _ = co.geodetic2spherical(
    #                 phi=lat_chunk, 
    #                 lambd=lon_chunk, 
    #                 ellipsoid=self.ellipsoid,
    #                 height=height_chunk
    #             )
    #             _lambda = np.radians(lon_chunk)
    #             m = np.arange(self.nmax + 1)
    #             mlambda = m[:, np.newaxis] * _lambda
    #             cosm = np.cos(mlambda)  # (nmax+1, chunk_size)
    #             sinm = np.sin(mlambda)  # (nmax+1, chunk_size)
                
    #             Pnm = ALFsGravityAnomaly(
    #                 vartheta=theta, 
    #                 nmax=self.nmax, 
    #                 ellipsoid=self.ellipsoid,
    #                 show_progress=leg_progress
    #             )
                
    #             H_flat[start:end] = (np.einsum('inm,nm,im->i', Pnm, self.HCnm, cosm.T) + 
    #                                 np.einsum('inm,nm,im->i', Pnm, self.HSnm, sinm.T))
        
    #     return H_flat.reshape(input_shape)

    # def dtm2006_height(
    #     self,
    #     lon: np.ndarray,
    #     lat: np.ndarray,
    #     chunk_size: int = 100,
    #     leg_progress: bool = False,
    #     height: np.ndarray = None
    # ) -> np.ndarray:
    #     '''
    #     Compute heights from DTM2006.0 spherical harmonic coefficients
        
    #     Parameters
    #     ----------
    #     lon         : geodetic longitude
    #     lat         : geodetic latitude
    #     chunk_size  : number of points to process at a time
    #     leg_progress: show progress bar for Legendre polynomial computation
    #     height      : height above ellipsoid (optional)
        
    #     Returns
    #     -------
    #     H           : synthesized height
    #     '''
    #     lon = np.asarray(lon)
    #     lat = np.asarray(lat)
        
    #     if lon.shape != lat.shape:
    #         raise ValueError('lon and lat must have the same shape')

    #     input_shape = lon.shape
    #     lon_flat = lon.flatten()
    #     lat_flat = lat.flatten()
    #     height_flat = np.zeros_like(lon_flat) if height is None else height.flatten()
    #     num_points = len(lon_flat)
        
    #     if num_points == 1:
    #         return self.dtm2006_height_point(lon_flat[0], lat_flat[0])

    #     H_flat = np.zeros(num_points)
        
    #     if num_points <= chunk_size:
    #         r, theta, _ = co.geodetic2spherical(
    #             phi=lat_flat, 
    #             lambd=lon_flat, 
    #             ellipsoid=self.ellipsoid,
    #             height=height_flat
    #         )
    #         _lambda = np.radians(lon_flat)  # (num_points,)
    #         m = np.arange(self.nmax + 1)    # (nmax+1,)
    #         mlambda = m[:, np.newaxis] * _lambda  # (nmax+1, num_points)
    #         cosm = np.cos(mlambda)                # (nmax+1, num_points)
    #         sinm = np.sin(mlambda)                # (nmax+1, num_points)
            
    #         Pnm = ALFsGravityAnomaly(
    #             vartheta=theta, 
    #             nmax=self.nmax, 
    #             ellipsoid=self.ellipsoid,
    #             show_progress=leg_progress
    #         )  # (num_points, nmax+1, nmax+1)
            
    #         H_flat = compute_harmonic_sum(Pnm, self.HCnm, self.HSnm, cosm, sinm)
    #     else:
    #         for start in tqdm(range(0, num_points, chunk_size), desc='Computing chunks'):
    #             end = min(start + chunk_size, num_points)
    #             lon_chunk = lon_flat[start:end]
    #             lat_chunk = lat_flat[start:end]
    #             height_chunk = height_flat[start:end]
                
    #             r, theta, _ = co.geodetic2spherical(
    #                 phi=lat_chunk, 
    #                 lambd=lon_chunk, 
    #                 ellipsoid=self.ellipsoid,
    #                 height=height_chunk
    #             )
    #             _lambda = np.radians(lon_chunk)  # (chunk_size,)
    #             m = np.arange(self.nmax + 1)     # (nmax+1,)
    #             mlambda = m[:, np.newaxis] * _lambda  # (nmax+1, chunk_size)
    #             cosm = np.cos(mlambda)                # (nmax+1, chunk_size)
    #             sinm = np.sin(mlambda)                # (nmax+1, chunk_size)
                
    #             Pnm = ALFsGravityAnomaly(
    #                 vartheta=theta, 
    #                 nmax=self.nmax, 
    #                 ellipsoid=self.ellipsoid,
    #                 show_progress=leg_progress
    #             )  # (chunk_size, nmax+1, nmax+1)
                
    #             H_flat[start:end] = compute_harmonic_sum(Pnm, self.HCnm, self.HSnm, cosm, sinm)
        
    #     return H_flat.reshape(input_shape)
    
    def dtm2006_height(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        chunk_size: int = 800,
        leg_progress: bool = False,
        height: np.ndarray = None,
        n_workers: int = None
    ) -> np.ndarray:
        '''
        Compute heights from DTM2006.0 spherical harmonic coefficients
        
        Parameters
        ----------
        lon         : geodetic longitude
        lat         : geodetic latitude
        chunk_size  : number of points to process at a time
        leg_progress: show progress bar for Legendre polynomial computation
        height      : height above ellipsoid (optional)
        n_workers   : number of parallel workers
        
        Returns
        -------
        H           : synthesized height
        '''
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        
        if lon.shape != lat.shape:
            raise ValueError('lon and lat must have the same shape')

        input_shape = lon.shape
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        height_flat = np.zeros_like(lon_flat) if height is None else height.flatten()
        num_points = len(lon_flat)
        
        # Check if self has the HCnm attribute
        if not hasattr(self, 'HCnm') or not hasattr(self, 'HSnm'):
            self.read_dtm2006()
        
        if num_points == 1:
            return self.dtm2006_height_point(lon_flat[0], lat_flat[0])

        H_flat = np.zeros(num_points)
        
        # Memory-based chunk size cap (optional)
        max_chunk_memory = 4e9  # ~4GB in bytes
        point_memory = (self.nmax + 1) ** 2 * 8
        max_points = int(max_chunk_memory / point_memory)
        chunk_size = min(chunk_size, max_points)
        
        if num_points <= chunk_size:
            r, theta, _ = co.geodetic2spherical(
                phi=lat_flat,
                lambd=lon_flat,
                ellipsoid=self.ellipsoid,
                height=height_flat
            )
            _lambda = np.radians(lon_flat)
            m = np.arange(self.nmax + 1)
            mlambda = m[:, np.newaxis] * _lambda
            cosm = np.cos(mlambda)
            sinm = np.sin(mlambda)
            
            Pnm = ALFsGravityAnomaly(
                vartheta=theta,
                nmax=self.nmax,
                ellipsoid=self.ellipsoid,
                show_progress=leg_progress
            )
            
            H_flat = compute_harmonic_sum(Pnm, self.HCnm, self.HSnm, cosm, sinm)
        else:
            n_workers = n_workers or cpu_count()
            chunk_starts = list(range(0, num_points, chunk_size))
            chunks = [
                (
                    self,
                    lon_flat[start:min(start + chunk_size, num_points)],
                    lat_flat[start:min(start + chunk_size, num_points)],
                    height_flat[start:min(start + chunk_size, num_points)],
                    self.nmax,
                    self.ellipsoid,
                    leg_progress,
                    i  # Chunk index for timing
                )
                for i, start in enumerate(chunk_starts)
            ]
            
            with Pool(processes=n_workers) as pool:
                results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc='Computing chunks'))
            
            for start, result in zip(chunk_starts, results):
                end = min(start + chunk_size, num_points)
                H_flat[start:end] = result
        
        return H_flat.reshape(input_shape)