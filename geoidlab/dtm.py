############################################################
# Utilities for digital terrain modeling                   #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
# import os
# os.environ['OMP_MAX_ACTIVE_LEVELS'] = '2'  # Set this at the very beginning
# os.environ['OMP_DISPLAY_ENV'] = 'FALSE'    # Optional: disable OpenMP environment display

import lzma
import os

import numpy as np

from pathlib import Path
from geoidlab import coordinates as co
from geoidlab import curtin
from geoidlab.legendre import ALFsGravityAnomaly
from geoidlab.numba.dtm import compute_harmonic_sum
from numba import get_num_threads

HIGH_DEGREE_STANDARD_WARNING = 2040



class DigitalTerrainModel:
    @staticmethod
    def _recommended_chunk_memory_bytes() -> int:
        '''
        Pick a conservative default chunk-memory budget that scales with
        available RAM when the user has not set an explicit limit.
        '''
        fallback = 2 * 1024**3
        try:
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (AttributeError, OSError, ValueError):
            return fallback

        # Aim for about 10% of system RAM by default. Keep a modest floor for
        # automatic mode only so very small systems still have a workable value.
        target = int(total_memory * 0.10)
        lower_bound = 1 * 1024**3
        return max(lower_bound, target)

    def __init__(
        self,
        model_name=None,
        nmax=2190,
        ellipsoid='wgs84',
        model_format=None,
        threaded_legendre: bool = False,
        legendre_method: str = 'standard',
        chunk_memory_gb: float | None = None,
    ) -> None:
        '''
        Initialize the DigitalTerrainModel class
        
        Parameters
        ----------
        model_name : Name of the DTM model file (full path)
        nmax       : Maximum degree of spherical harmonics
        ellipsoid  : Reference ellipsoid
        model_format: Optional explicit model format.
                      Supported: 'dtm2006_text', 'bshc'

        Returns
        -------
        None
        '''
        self.name = model_name
        self.nmax = nmax
        self.ellipsoid = ellipsoid
        self.model_format = model_format
        self.threaded_legendre = threaded_legendre
        self.legendre_method = legendre_method
        self.chunk_memory_gb = chunk_memory_gb
        # self.progress = show_progress

        if self.legendre_method == 'standard' and self.nmax >= HIGH_DEGREE_STANDARD_WARNING:
            print(
                f"Warning: nmax={self.nmax} with legendre_method='standard' may show numerical artefacts "
                f"at high degree. If you notice latitude-band artefacts, rerun with "
                f"legendre_method='holmes'."
            )

        if self.name is None:
            script_dir: Path = Path(__file__).resolve().parent
            self.name = script_dir / 'data' / 'DTM2006.xz'
            self.model_format = self.model_format or 'dtm2006_text'
            print(f'Using compressed DTM2006.0 file in {script_dir}/data ...')
            if self.nmax < 2190:
                print('Note: Maximum degree for DTM2006.0 is 2190') 
            with lzma.open(self.name, 'rt') as f:
                self.dtm = f.readlines()
        else:
            self.name = Path(self.name)
            self.model_format = self.model_format or self._infer_model_format(self.name)
            try:
                print(f'Reading DTM file {self.name} ...')
                if self.model_format == 'dtm2006_text' and self.name.suffix == '.xz':
                    with lzma.open(self.name, 'rt') as f:
                        self.dtm = f.readlines()
                elif self.model_format == 'dtm2006_text':
                    with open(self.name, 'r') as f:
                        self.dtm = f.readlines() # self.dtm is the DTM2006 text file
                elif self.model_format == 'bshc':
                    # Binary format: read lazily in read_coefficients().
                    pass
                else:
                    raise ValueError(
                        f'Unsupported model_format: {self.model_format}. '
                        "Supported: 'dtm2006_text', 'bshc'"
                    )
            except Exception as e:
                raise Exception(f'Error reading DTM file {self.name}: {str(e)}')
                
    @staticmethod
    def _infer_model_format(filename: Path) -> str:
        suffix = filename.suffix.lower()
        if suffix == '.bshc':
            return 'bshc'
        return 'dtm2006_text'

    def _synthesize_chunk(
        self,
        lon_chunk: np.ndarray,
        lat_chunk: np.ndarray,
        height_chunk: np.ndarray,
        leg_progress: bool = False,
    ) -> np.ndarray:
        _, theta, _ = co.geodetic2spherical(
            phi=lat_chunk,
            lambd=lon_chunk,
            ellipsoid=self.ellipsoid,
            height=height_chunk,
        )
        _lambda = np.radians(lon_chunk)
        m = np.arange(self.nmax + 1)
        mlambda = m[:, np.newaxis] * _lambda
        cosm = np.cos(mlambda)
        sinm = np.sin(mlambda)
        
        Pnm, _ = ALFsGravityAnomaly(
            vartheta=theta,
            nmax=self.nmax,
            ellipsoid=self.ellipsoid,
            show_progress=leg_progress,
            threaded=self.threaded_legendre,
            backend=self.legendre_method,
        )
        return compute_harmonic_sum(Pnm, self.HCnm, self.HSnm, cosm, sinm)

    def read_dtm2006(self) -> None:
        '''
        Backward-compatible reader for DTM2006 text format.

        For format-agnostic loading, use `read_coefficients()`.
        '''
        if self.model_format != 'dtm2006_text':
            raise ValueError(
                f'read_dtm2006 is only valid for dtm2006_text format, got {self.model_format}'
            )

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

    def read_coefficients(self) -> None:
        '''
        Read spherical harmonic coefficients for the configured potential model.
        '''
        if hasattr(self, 'HCnm') and hasattr(self, 'HSnm'):
            return

        if self.model_format == 'dtm2006_text':
            self.read_dtm2006()
            return

        if self.model_format == 'bshc':
            if curtin.is_potential_model_filename(self.name):
                raise ValueError(
                    f'{self.name.name} is a Curtin dV_* potential model, not a relief-height SHC model. '
                    'Use Earth2014 SHC files such as SUR2014, BED2014, ICE2014, RET2014, or TBI2014 '
                    'when synthesizing topography or bathymetry with DigitalTerrainModel.'
                )
            coeffs = curtin.read_bshc_coefficients(self.name, nmax=self.nmax)
            self.HCnm = coeffs['HCnm']
            self.HSnm = coeffs['HSnm']
            if not np.isfinite(self.HCnm).all() or not np.isfinite(self.HSnm).all():
                raise ValueError(
                    f'Loaded non-finite spherical harmonic coefficients from {self.name}.'
                )
            return

        raise ValueError(
            f'Unsupported model_format: {self.model_format}. '
            "Supported: 'dtm2006_text', 'bshc'"
        )
    
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
            self.read_coefficients()
            
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
        
        Pnm, _ = ALFsGravityAnomaly(
            vartheta=np.array([theta]),
            nmax=self.nmax,
            ellipsoid=self.ellipsoid,
            show_progress=False,
            threaded=self.threaded_legendre,
            backend=self.legendre_method,
        )
        H = np.sum((self.HCnm * Pnm[0]) @ cosm + (self.HSnm * Pnm[0]) @ sinm)
        
        return float(H)

    def synthesize_height_point(
        self,
        lon: float,
        lat: float,
        height=None
    ) -> float:
        '''
        Compute height for a single point using the configured spherical harmonic terrain model.

        Notes
        -----
        This is a model-agnostic alias for `dtm2006_height_point`, so the same API can be used
        with DTM2006 text models and Curtin Earth2014 `.bshc` models.
        '''
        return self.dtm2006_height_point(lon=lon, lat=lat, height=height)
    
    def dtm2006_height(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        chunk_size: int = 800,
        leg_progress: bool = False,
        height: np.ndarray = None,
        n_workers: int = None,
        save: bool = True
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
        n_workers   : retained for API compatibility; chunk execution now stays in one process
        
        Returns
        -------
        H           : synthesized height
        '''
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        
        if lon.shape != lat.shape:
            raise ValueError('lon and lat must have the same shape')

        input_shape = lon.shape
        lon_flat = lon.ravel()
        lat_flat = lat.ravel()
        height_flat = np.zeros_like(lon_flat) if height is None else height.ravel()
        num_points = len(lon_flat)
        
        # Check if self has the HCnm attribute
        if not hasattr(self, 'HCnm') or not hasattr(self, 'HSnm'):
            self.read_coefficients()
        
        if num_points == 1:
            return self.dtm2006_height_point(lon_flat[0], lat_flat[0])

        H_flat = np.zeros(num_points)
        
        requested_chunk_size = chunk_size

        # Memory-based chunk size cap (optional). Account for the full ALF cube
        # plus the packed Holmes-style workspace used by the stable backend.
        if self.chunk_memory_gb is not None:
            max_chunk_memory = int(self.chunk_memory_gb * 1024**3)
        else:
            max_chunk_memory = self._recommended_chunk_memory_bytes()
        point_memory = ((self.nmax + 1) ** 2 + ((self.nmax + 1) * (self.nmax + 2) // 2)) * 8
        max_points = int(max_chunk_memory / point_memory)
        chunk_size = min(chunk_size, max(1, max_points))
        
        n_chunks = (num_points + chunk_size - 1) // chunk_size
        print(
            f'Data will be processed in {n_chunks} chunks using up to {get_num_threads()} '
            f'Numba threads (Legendre mode: '
            f"{'standard-threaded' if self.legendre_method == 'standard' and self.threaded_legendre else self.legendre_method}).\n"
        )
        if chunk_size < num_points:
            print(
                f'Effective DTM chunk size: {chunk_size} points '
                f'(requested: {requested_chunk_size}, '
                f'memory budget ~{max_chunk_memory / 1024**3:.1f} GiB).\n'
            )

        for i, start in enumerate(range(0, num_points, chunk_size)):
            end = min(start + chunk_size, num_points)
            print(f'Processing chunk {i + 1} of {n_chunks}...')
            H_flat[start:end] = self._synthesize_chunk(
                lon_flat[start:end],
                lat_flat[start:end],
                height_flat[start:end],
                leg_progress=leg_progress and i == 0,
            )
        
        H = H_flat.reshape(input_shape)
        
        if save:
            DigitalTerrainModel.save_dtm2006_height(lon, lat, H, self.nmax)
            
        return H

    def synthesize_height(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        chunk_size: int = 800,
        leg_progress: bool = False,
        height: np.ndarray = None,
        n_workers: int = None,
        save: bool = True
    ) -> np.ndarray:
        '''
        Compute heights using the configured spherical harmonic terrain model.

        Notes
        -----
        This is a model-agnostic alias for `dtm2006_height`, so the same API can be used
        with DTM2006 text models and Curtin Earth2014 `.bshc` models.
        '''
        return self.dtm2006_height(
            lon=lon,
            lat=lat,
            chunk_size=chunk_size,
            leg_progress=leg_progress,
            height=height,
            n_workers=n_workers,
            save=save
        )
    
    # Create a static method that saves writes H as a netcdf files if H is a 2D array
    @staticmethod
    def save_dtm2006_height(
        lon: np.ndarray,
        lat: np.ndarray,
        H: np.ndarray,
        nmax: int,
        filename: str = 'H_dtm2006.nc'
    ) -> None:
        '''
        Save synthesized height to a netCDF file
        
        Parameters
        ----------
        lon       : 2D array of geodetic longitudes
        lat       : 2D array of geodetic latitudes
        H         : 2D array of synthesized heights
        filename  : path to the netCDF file
        '''
        if nmax is None:
            raise ValueError('nmax must be provided')
        
        from netCDF4 import Dataset
        from datetime import datetime, timezone
        
        # Ensure all inputs are 2D arrays with the same shape
        if lon.shape != lat.shape or lon.shape != H.shape:
            raise ValueError('lon, lat, and H must have the same shape')
        
        # Create the output directory if it doesn't exist
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update filename to include nmax
        filename = filename.replace('.nc', f'_{nmax}.nc')
        
        # Full file path
        file_path = output_dir / filename
        
        # Extract 1D coordinate arrays from 2D grids (assuming regular grid from np.meshgrid)
        lon_1d = lon[0, :]
        lat_1d = lat[:, 0]
        
        # Create and write to the netCDF file
        with Dataset(file_path, 'w', format='NETCDF4') as ds:
            # --- Global Attributes ---
            ds.title = 'Synthesized Heights from DTM2006.0 Model'
            ds.description = (
                'This dataset contains synthesized heights representing terrain elevations '
                'computed from the DTM2006.0 spherical harmonic model.'
            )
            ds.creation_date = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            ds.source = 'DTM2006.0 spherical harmonic coefficients'
            ds.author = 'geoidlab development team'  
            ds.software = 'geoidlab'  
            ds.conventions = 'CF-1.8'  # Optional, aligns with geospatial standards
            
            # --- Dimensions ---
            ds.createDimension('lon', len(lon_1d))
            ds.createDimension('lat', len(lat_1d))
            
            # --- Variables ---
            # Longitude
            lon_var = ds.createVariable('lon', 'f8', ('lon',))
            lon_var.long_name = 'longitude'
            lon_var.units = 'degrees'
            lon_var.standard_name = 'longitude'
            lon_var.description = 'Geodetic longitude of grid points, measured east from Greenwich.'
            
            # Latitude
            lat_var = ds.createVariable('lat', 'f8', ('lat',))
            lat_var.long_name = 'latitude'
            lat_var.units = 'degrees'
            lat_var.standard_name = 'latitude'
            lat_var.description = 'Geodetic latitude of grid points, measured north from the equator.'
            
            # Synthesized Height
            H_var = ds.createVariable('H', 'f8', ('lat', 'lon'))
            H_var.long_name = 'synthesized_height'
            H_var.units = 'meters'
            H_var.standard_name = 'height_above_reference_ellipsoid' 
            H_var.description = (
                'Synthesized terrain height above the reference ellipsoid, computed from '
                'DTM2006.0 spherical harmonic coefficients.'
            )
            H_var.coordinates = 'lon lat'  # Link to coordinate variables
            
            # --- Write Data ---
            lon_var[:] = lon_1d
            lat_var[:] = lat_1d
            H_var[:, :] = H
