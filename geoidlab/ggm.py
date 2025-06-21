############################################################
# Utilities for synthesizing gravity functionals           #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from pathlib import Path

from geoidlab import(
    icgem, 
    constants, 
    gravity, 
    coordinates as co, 
    shtools
    ) 


from geoidlab.legendre import ALFsGravityAnomaly, ALF
from geoidlab.numba.ggm import (
    compute_gravity_chunk, 
    compute_disturbance_chunk,
    compute_disturbing_potential_chunk,
    compute_second_radial_chunk,
    compute_separation_chunk,
    compute_disturbing_potential_derivative_chunk
)

from tqdm import tqdm
from IPython.display import clear_output

import numpy as np
import pandas as pd


class GlobalGeopotentialModel:
    def __init__(
        self, 
        shc=None, 
        model_name=None, 
        ellipsoid='wgs84', 
        nmax=90, 
        grav_data=None, 
        zonal_harmonics=True,
        model_dir='downloads',
        chunk_size=None,
        dtm_model=None
    ) -> None:
        '''
        Initialize the GlobalGeopotentialModel class for potential modeling
        
        Parameters
        ----------
        shc             : Spherical Harmonic Coefficients -- output of icgem.read_icgem()
        model_name      : Name of gravity model without extension
        ellipsoid       : Reference ellipsoid ('wgs84' or 'grs80')
        nmax            : Maximum spherical harmonic degree
        grav_data       : Gravity data with columns lon, lat, and elevation: lat and lon units: degrees
        zonal_harmonics : Whether to subtract even zonal harmonics or not
        model_dir       : Directory where model file is stored
        chunk_size      : Size of chunks for parallel processing (default: 100)
        dtm_model       : Digital Terrain Model (DTM) to use for topographic effects (optional)
        '''
        self.shc       = shc
        self.model     = model_name
        self.ellipsoid = ellipsoid
        self.nmax      = nmax
        self.grav_data = grav_data
        self.model_dir = model_dir
        self.chunk     = chunk_size
        self.dtm_model = dtm_model
        
        # Input validation
        if self.model_dir is None:
            self.model_dir = 'downloads'

        if self.model.endswith('.gfc'):
            self.model = self.model.split('.gfc')[0]
            
        if self.shc is None and self.model is None:
            raise ValueError('Either shc or model_name must be specified')
        
        if self.chunk is None:
            print('Chunk size not specified. Setting chunk size to 100...')
            self.chunk = 100
            
        if self.dtm_model is not None:
            self.dtm_model = Path(self.dtm_model)
            if not self.dtm_model.exists():
                print(f'Warning: DTM model {self.dtm_model} does not exist. Using default DTM2006.')
                self.dtm_model = None
            else:
                print(f'User-specified DTM model exists and will be used: {self.dtm_model}')
            
        try:
            if self.shc is None:
                self.shc = icgem.read_icgem(str(Path(self.model_dir) / (self.model + '.gfc')))
        except Exception as e:
            raise ValueError(f'Failed to read model file: {e}')
        
        # Subtract even zonal harmonics
        self.shc = shtools.subtract_zonal_harmonics(self.shc, self.ellipsoid) if zonal_harmonics else self.shc
        
        if self.grav_data is None:
            raise ValueError('Provide data with columns lon, lat, and elevation in order')
        elif isinstance(self.grav_data, str):
            self.grav_data = self.read_file()
        else:
            if isinstance(self.grav_data, np.ndarray):
                self.lon = self.grav_data[:,0]
                self.lat = self.grav_data[:,1]
                try:
                    self.h = self.grav_data[:,2]
                except IndexError:
                    print('Looks like there is no elevation column. Setting elevation to 0')
                    self.h = np.zeros(len(self.lat))
            elif isinstance(self.grav_data, pd.DataFrame):
                lon_column = [col for col in self.grav_data.columns if pd.Series(col).str.contains('lon', case=False).any()][0]
                lat_column = [col for col in self.grav_data.columns if pd.Series(col).str.contains('lat', case=False).any()][0]
                self.lon = self.grav_data[lon_column].values
                self.lat = self.grav_data[lat_column].values
                try:
                    elev_column = [col for col in self.grav_data.columns if pd.Series(col).str.contains(r'elev|height', case=False).any()][0]
                    self.h = self.grav_data[elev_column].values
                except IndexError:
                    print('Looks like there is no elevation column. Setting elevation to 0')
                    self.h = np.zeros(len(self.lat))
        self.r, self.vartheta, _ = co.geodetic2spherical(phi=self.lat, lambd=self.lon, height=self.h, ellipsoid=self.ellipsoid)
        
        # Set self.r = self.shc['a'] if all self.h = 0 (Doesn't really make a big difference)
        if np.all(self.h == 0):
            print('Setting r = R ...')
            self.r = self.shc['a'] * np.ones(len(self.lon))
            
        self.lambda_ = np.radians(self.lon)
        # self.phi     = np.radians(self.lat)


    def gravity_anomaly_sequential(self) -> np.ndarray:
        '''
        Compute long-wavelength gravity anomalies from a GGM
        
        Returns
        -------
        Dg     : Gravity anomalies [mGal]
        
        Reference
        ---------
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        '''
        Pnm, _ = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        Dg  = np.zeros(len(self.lon))
        
        for n in tqdm(range(2, self.nmax + 1), desc='Computing gravity anomalies'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            Dg += (n - 1) * (self.shc['a'] / self.r) ** n * sum
            
        return self.shc['GM'] / self.r ** 2 * Dg * 10**5 # [mGal]
    
    def gravity_anomaly_parallel(self) -> np.ndarray:
        '''
        Compute gravity anomalies using chunking and Numba optimization.
        
        Returns
        -------
        Dg : Gravity anomaly array (mGal)
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a = np.array(self.shc['a'])
        GM = np.array(self.shc['GM'])
        
        Dg = np.zeros(len(self.lon))
        
        # if self.chunk is None or self.chunk >= len(self.lon):
        #     # Process all points at once
        #     lon_rad = np.radians(self.lon)
        #     Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
        #     for n in tqdm(range(2, self.nmax + 1), desc='Computing gravity anomalies'):
        #         Dg += compute_gravity_chunk(Cnm, Snm, lon_rad, a, self.r, Pnm, n)
        # else:
            # Process in chunks
        n_points = len(self.lon)
        n_chunks = (n_points // self.chunk) + 1
        print(f'Data will be processed in {n_chunks} chunks...\n')

        for i in range(n_chunks):
            start_idx = i * self.chunk
            end_idx = min((i + 1) * self.chunk, n_points)
            
            lon_chunk = self.lon[start_idx:end_idx]
            r_chunk = self.r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            Dg_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)  # Clear the output in Jupyter Notebook
            # print(f'Processing chunk {i + 1} of {n_chunks}...', end='\r') # for notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...')
            Pnm_chunk, _ = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                Dg_chunk += compute_gravity_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)
            
            Dg[start_idx:end_idx] = Dg_chunk
            # print('\n') # for Jupyter Notebook
        
        return GM / self.r ** 2 * Dg * 10**5  # Convert to mGal
    
    def gravity_anomaly(self, parallel: bool=True, batch_size: int = 1000) -> np.ndarray:
        '''
        Method to call either gravity_anomaly_parallel() or gravity_anomaly_sequential()
        with automatic batching for large arrays
        
        Parameters
        ----------
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        Dg        : Gravity anomaly array (mGal)
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('gravity_anomaly', batch_size=batch_size, parallel=parallel)
        
        # Original implementation for smaller arrays
        if parallel:
            return self.gravity_anomaly_parallel()
        else:
            return self.gravity_anomaly_sequential()
        
    
    def gravity_disturbance_sequential(self) -> np.ndarray:
        '''
        Compute long-wavelength gravity disturbance from a GGM
        
        Returns
        -------
        dg     : Gravity disturbance [mGal]
        
        Reference
        ---------
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        '''
        
        Pnm, _ = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        dg  = np.zeros(len(self.lon))
        
        for n in tqdm(range(2, self.nmax + 1), desc='Computing gravity anomalies'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            dg += (n + 1) * (self.shc['a'] / self.r)**n * sum
                
        return self.shc['GM'] / self.r ** 2 * dg * 10**5 # [mGal]
    
    def gravity_disturbance_parallel(self) -> np.ndarray:
        '''
        Compute gravity disturbance using chunking and Numba optimization.
        
        Returns
        -------
        dg : Gravity disturbance array (mGal)
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a = np.array(self.shc['a'])
        GM = np.array(self.shc['GM'])
        
        dg = np.zeros(len(self.lon))
        
        n_points = len(self.lon)
        n_chunks = (n_points // self.chunk) + 1
        print(f'Data will be processed in {n_chunks} chunks...\n')

        for i in range(n_chunks):
            start_idx = i * self.chunk
            end_idx = min((i + 1) * self.chunk, n_points)
            
            lon_chunk = self.lon[start_idx:end_idx]
            r_chunk = self.r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            dg_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)  # Clear the output in Jupyter notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...') #, end='\r')
            Pnm_chunk, _ = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                dg_chunk += compute_disturbance_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)

            dg[start_idx:end_idx] = dg_chunk
            # print('\n')
        
        return GM / self.r ** 2 * dg * 10**5  # [mGal]
    
    def gravity_disturbance(self, parallel: bool=True, batch_size: int = 1000) -> np.ndarray:
        '''
        Method to call either gravity_disturbance_parallel() or gravity_disturbance_sequential()
        with automatic batching for large arrays
        
        Parameters
        ----------
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        dg         : Gravity disturbance array (mGal)
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('gravity_disturbance', batch_size=batch_size, parallel=parallel)
        
        # Original implementation for smaller arrays
        if parallel:
            return self.gravity_disturbance_parallel()
        else:
            return self.gravity_disturbance_sequential()
        
        
    def disturbing_potential_sequential(self, r_or_R='r') -> np.ndarray:
        '''
        Compute long-wavelength disturbing potential from a GGM
        
        Parameters
        ----------
        r_or_R    : Either 'r' for radial distance or 'R' for reference radius
        Returns
        -------
        T         : Disturbing potential [m2/s2]
        
        Notes
        -----
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        2. Please ensure that you have called shtools.subtract_zonal_harmonics() on shc before passing it to disturbing_potential()
        3. `r_or_R` parameter is used to calculate T for geoid heights (R) or height anomalies (r). See last paragraph on Page 296
        '''
        r = self.shc['a'] * np.ones(len(self.lon)) if r_or_R == 'R' else self.r
        
        Pnm, _ = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        T   = np.zeros(len(self.lon))
        
        for n in tqdm(range(2, self.nmax + 1), desc='Computing disturbing potential'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            T += (self.shc['a'] / r)**n * sum
        
        T = self.shc['GM'] / r * T # [m2/s2]
        
        return T
    
    def disturbing_potential_parallel(self, r_or_R='r') -> np.ndarray:
        '''
        Compute disturbing potential using chunking and Numba optimization.
        
        Parameters
        ----------
        r_or_R    : Either 'r' for radial distance or 'R' for reference radius
        Returns
        -------
        T         : Disturbing potential array (m2/s2)
        '''
    
        r = self.shc['a'] * np.ones(len(self.lon)) if r_or_R == 'R' else self.r

        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a = np.array(self.shc['a'])
        GM = np.array(self.shc['GM'])
        
        T = np.zeros(len(self.lon))
                
        n_points = len(self.lon)
        n_chunks = (n_points // self.chunk) + 1
        
        for i in range(n_chunks):
            start_idx = i * self.chunk
            end_idx = min((i + 1) * self.chunk, n_points)
            
            lon_chunk = self.lon[start_idx:end_idx]
            r_chunk = r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            T_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)  # Clear the output in Jupyter notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...') #, end='\r')
            Pnm_chunk, _ = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                T_chunk += compute_disturbing_potential_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)

            T[start_idx:end_idx] = T_chunk
            # print('\n')
        
        T = GM / r * T # [m2/s2]
        
        return T
    
    
    def disturbing_potential(self, r_or_R='r', parallel: bool=True, batch_size: int = 1000) -> np.ndarray:
        '''
        Method to call either disturbing_potential_parallel() or disturbing_potential_sequential()
        with automatic batching for large arrays
        
        Parameters
        ----------
        r_or_R    : Either 'r' for radial distance or 'R' for reference radius
        parallel  : Whether to use parallel processing (forced to False for large arrays)
        batch_size: Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        T         : Disturbing potential array (m2/s2)
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('disturbing_potential', batch_size=batch_size, 
                                     r_or_R=r_or_R, parallel=parallel)
        
        # Original implementation for smaller arrays
        if parallel:
            return self.disturbing_potential_parallel(r_or_R=r_or_R)
        else:
            return self.disturbing_potential_sequential(r_or_R=r_or_R)
    
    def second_radial_derivative_sequential(self) -> np.ndarray:
        '''
        Compute second radial derivative (vertical gradient) from a GGM
        
        Returns
        -------
        Tzz     : Second radial derivative [Eötvös (E): 10^{−9}s^{−2}]
        
        Reference
        ---------
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.39, p.298
        '''
        Pnm, _ = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        Tzz = np.zeros(len(self.lon))
        
        for n in tqdm(range(2, self.nmax + 1), desc='Computing gravity anomalies'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            Tzz += (n + 1) * (n+2) * (self.shc['a'] / self.r)**n * sum
                
        return self.shc['GM'] / self.r ** 3 * Tzz * 10**9 # [E = Eötvös]
    
    
    def second_radial_derivative_parallel(self) -> np.ndarray:
        '''
        Compute second radial derivative using chunking and Numba optimization.
        
        
        Returns
        -------
        Tzz    : Second radial derivative (Eötvös (E): 10^{−9}s^{−2})
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a = np.array(self.shc['a'])
        GM = np.array(self.shc['GM'])
        
        Tzz = np.zeros(len(self.lon))
                
        n_points = len(self.lon)
        n_chunks = (n_points // self.chunk) + 1
        
        for i in range(n_chunks):
            start_idx = i * self.chunk
            end_idx = min((i + 1) * self.chunk, n_points)
            
            lon_chunk = self.lon[start_idx:end_idx]
            r_chunk = self.r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            Tzz_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)  # Clear the output in Jupyter notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...') #, end='\r')
            Pnm_chunk, _ = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                Tzz_chunk += compute_second_radial_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)

            Tzz[start_idx:end_idx] = Tzz_chunk
            # print('\n')
        
        
        return GM / self.r ** 3 * Tzz * 10**9 # [E = Eötvös]
    
    def second_radial_derivative(self, parallel: bool=True, batch_size: int = 1000) -> np.ndarray:
        '''
        Method to call either second_radial_derivative_parallel() or second_radial_derivative_sequential()
        with automatic batching for large arrays
        
        Parameters
        ----------
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        Tzz        : Second radial derivative (Eötvös (E): 10^{−9}s^{−2})
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('second_radial_derivative', batch_size=batch_size, parallel=parallel)
        
        # Original implementation for smaller arrays
        if parallel:
            return self.second_radial_derivative_parallel()
        else:
            return self.second_radial_derivative_sequential()
        
    def height_anomaly(
        self, 
        T=None, 
        tolerance=5e-3, 
        max_iter=5,
        parallel: bool=True,
        batch_size: int = 1000
    ) -> np.ndarray:
        '''
        Height anomaly based on Bruns' method with automatic batching for large arrays
        
        Parameters
        ----------
        T          : Disturbing potential array (m2/s2)
        tolerance  : Tolerance for refining height anomaly
        max_iter   : Maximum number of iterations
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        zeta       : Height anomaly (m)
        
        Notes
        -----
        1. Torge, Muller, & Pail (2023): Geodesy, Eq. 6.9, p.288
        2. For arrays larger than batch_size, automatic batching is used to prevent memory issues
        3. We need to iterate to refine zeta because we start estimating zeta using `H` not `hn`
            Steps:
                1. Estimate hn (normal height): hn = self.h - zeta
                2. Use hn to calculate gammaQ
                3. Use gammaQ to calculate updated zeta (zeta1)
                4. Repeat until convergence.
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('height_anomaly', batch_size=batch_size, 
                                     T=T, tolerance=tolerance, max_iter=max_iter, parallel=parallel)
        
        # Original implementation for smaller arrays
        print('Using Bruns\' method with zero-degree correction to calculate height anomaly...\n')
        
        T = self.disturbing_potential(r_or_R='r', parallel=parallel) if T is None else T
        gammaQ = gravity.normal_gravity_above_ellipsoid(phi=self.lat, h=self.h, ellipsoid=self.ellipsoid)
        gammaQ = gammaQ * 1e-5  # mgal to m/s2
        zeta_old = T / gammaQ
        zeta_old = self.zero_degree_term(geoid=zeta_old, zeta_or_geoid='zeta')
        
        if max_iter == 0:
            return zeta_old
        
        print(f'Iterating to refine zeta...\n')
        
        converged = np.zeros_like(zeta_old, dtype=bool)
        
        for _ in range(max_iter):
            print(f'Iteration {_ + 1}')
            gammaQ = gravity.normal_gravity_above_ellipsoid(phi=self.lat, h=self.h - zeta_old, ellipsoid=self.ellipsoid)
            gammaQ = gammaQ * 1e-5  # mgal to m/s2
            zeta = T / gammaQ
            
            # Zero-degree term
            zeta = self.zero_degree_term(geoid=zeta, zeta_or_geoid='zeta')
            
            zeta_diff = np.abs(zeta - zeta_old)
            
            # Update the converged mask
            converged = converged | (zeta_diff < tolerance)
            
            # If all points have converged, break the loop
            if np.all(converged):
                break
            
            # Update zeta_old only for points that have not converged
            zeta_old[~converged] = zeta[~converged]
        
        return zeta
    
    def geoid(self, T=None, icgem: bool = False, parallel: bool = False, batch_size: int = 1000) -> np.ndarray:
        '''
        Geoid heights based on Bruns' method with automatic batching for large arrays
        
        Parameters
        ----------
        T          : Disturbing potential array (m2/s2)
        icgem      : Use ICGEM formula which accounts for topographic effect
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        zeta       : Geoid height (m)
        
        Notes
        -----
        1. Torge, Muller, & Pail (2023): Geodesy, Eq. 6.8, p.288
        2. For arrays larger than batch_size, automatic batching is used to prevent memory issues
        3. ICGEM computes the reference geoid using two contributions:
                1. The geoid as computed by `geoid` method (N)
                2. The contribution of topography (N_topo)
                3. Finally, they obtain the geoid as N = N - N_topo
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('geoid', batch_size=batch_size, T=T, icgem=icgem, parallel=parallel)
        
        # Original implementation for smaller arrays
        print('Using Bruns\' method with zero-degree correction to calculate geoid height...\n')

        T = self.disturbing_potential(r_or_R='R', parallel=parallel) if T is None else T
        gamma0 = gravity.normal_gravity_somigliana(phi=self.lat, ellipsoid=self.ellipsoid)

        N = T / gamma0

        # Remove topographic effect if requested
        N_topo = np.zeros_like(N)
        if icgem:
            if self.dtm_model is None:
                print('ICGEM version requested. Computing topographic contribution to geoid height (N_topo) using DTM2006')
            else:
                print(f'ICGEM version requested. Computing topographic contribution to geoid height (N_topo) using {self.dtm_model.stem}')
            from geoidlab.dtm import DigitalTerrainModel
            dtm = DigitalTerrainModel(nmax=self.nmax, ellipsoid=self.ellipsoid, model_name=self.dtm_model)
            H = dtm.dtm2006_height(lon=self.lon, lat=self.lat, chunk_size=self.chunk, save=False)
            N_topo = 2 * np.pi * constants.earth()['G'] * constants.earth()['rho'] * H ** 2 / gamma0
            print('Subtracting topographic effect from geoid height...')
        
        N -= N_topo
        
        # Zero-degree term
        N = self.zero_degree_term(geoid=N, zeta_or_geoid='geoid')
        print('Reference geoid computed successfully!')
        
        return N

    def zero_degree_term(self, geoid=None, GM=None, zeta_or_geoid='geoid') -> np.ndarray:
        '''
        Add zero-degree term to the GGM geoid
        
        Parameters
        ----------
        geoid         : Geoid model (output of ggm.reference_geoid())
        GM            : Gravity constant of the GGM
        zeta_or_geoid : 'zeta' or 'geoid'
        Returns
        -------
        N             : Geoid corrected for zero-degree term
        
        Reference
        ---------
        Hofmann-Wellenhof & Moritz (2006): Physical Geodesy, Eq. 2–356, p. 113
        '''
        if geoid is None:
            raise ValueError('Please provide geoid')
        
        if self.shc is None and GM is None:
            raise ValueError('Please provide shc (output of icgem.read_icgem()) or GM from GGM')
        
        if self.shc is not None and GM is None:
            GM = self.shc['GM']
        
        ref_ellipsoid = constants.wgs84() if 'wgs84' in self.ellipsoid.lower() else constants.grs80()
        GM0 = ref_ellipsoid['GM']
        U0  = ref_ellipsoid['U0'] # Potential of ellipsoid (m2/s2)
        R   = ref_ellipsoid['semi_major'] if zeta_or_geoid == 'geoid' else self.r
        
        W0  = constants.earth()['W0']
        
        if zeta_or_geoid == 'zeta':
            gamma   = gravity.normal_gravity_above_ellipsoid(
                phi=self.lat, h=self.h, ellipsoid=self.ellipsoid
            ) # This is actually gamma_Q, but we use gamma here for convenience
            gamma   = gamma   / 1e5 # mgal to m/s2
        else:
            gamma   = gravity.normal_gravity_somigliana(phi=self.lat, ellipsoid=self.ellipsoid)
        
        N = geoid + ( (GM - GM0) / R - (W0 - U0) ) / gamma   
        
        return N
    
    def separation_sequential(self) -> np.ndarray:
        '''
        Compute the geoid-quasi geoid separation from a GGM sequentially.

        Returns
        -------
        H      : Geoid-quasi geoid separation values.
        '''

        Pnm, _ = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        H = np.zeros(len(self.lon))

        for n in tqdm(range(0, self.nmax + 1), desc='Computing separation'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            H += sum

        return H

    def separation_parallel(self) -> np.ndarray:
        '''
        Compute the geoid-quasi geoid separation using chunking and Numba optimization.
        
        Returns
        -------
        H : Geoid-quasi geoid separation values.
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a = np.array(self.shc['a'])

        H = np.zeros(len(self.lon))
        n_points = len(self.lon)
        n_chunks = (n_points // self.chunk) + 1
        print(f'Data will be processed in {n_chunks} chunks...\n')

        for i in range(n_chunks):
            start_idx = i * self.chunk
            end_idx = min((i + 1) * self.chunk, n_points)
            
            lon_chunk = self.lon[start_idx:end_idx]
            r_chunk = self.r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            H_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)
            print(f'Processing chunk {i + 1} of {n_chunks}...', end='\r')
            Pnm_chunk, _ = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(0, self.nmax + 1):
                H_chunk += compute_separation_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)
            
            H[start_idx:end_idx] = H_chunk
            print('\n')
        
        return H
    
    def separation(self, parallel: bool=True, batch_size: int = 1000) -> np.ndarray:
        '''
        Method to call either separation_parallel() or separation_sequential()
        with automatic batching for large arrays
        
        Parameters
        ----------
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        H          : Geoid-quasi geoid separation values
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('separation', batch_size=batch_size, parallel=parallel)
        
        # Original implementation for smaller arrays
        if parallel:
            return self.separation_parallel()
        else:
            return self.separation_sequential()
        
        
    def read_file(self) -> pd.DataFrame:
        '''
        Read file containing gravity data (or lat/lon data)
        
        Returns
        -------
        df        : Pandas DataFrame
        '''
        column_mapping = {
            'lon': ['lon', 'long', 'longitude', 'x'],
            'lat': ['lat', 'lati', 'latitude', 'y'],
            'h': ['h', 'height', 'z', 'elevation', 'elev'],
            'gravity': ['gravity', 'g', 'acceleration', 'grav']
        }
        file_path = self.grav_data
        
        if file_path is None:
            raise ValueError('File path not specified')
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError('Unsupported file format')

        # Rename columns to standardized names
        df = df.rename(columns=lambda col: next((key for key, values in column_mapping.items() if col.lower() in values), col))

        # Ensure the DataFrame only contains the expected columns, fill missing ones with NaN
        expected_columns = ['lon', 'lat', 'h', 'gravity']
        df = df[[col for col in df.columns if col in expected_columns]]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        return df
    
    def disturbing_potential_derivative_sequential(self, r_or_R='r') -> np.ndarray:
        '''
        Compute the derivative of the disturbing potential with respect to theta (colatitude).
        
        Parameters
        ----------
        r_or_R    : Either 'r' for radial distance or 'R' for reference radius
        Returns
        -------
        dTdtheta  : Derivative of disturbing potential [m2/s2/rad]
        
        Notes
        -----
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        2. Please ensure that you have called shtools.subtract_zonal_harmonics() on shc before passing it to disturbing_potential()
        3. `r_or_R` parameter is used to calculate T for geoid heights (R) or height anomalies (r). See last paragraph on Page 296
        '''
        r = self.shc['a'] * np.ones(len(self.lon)) if r_or_R == 'R' else self.r
        
        _, dPnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        dTdtheta   = np.zeros(len(self.lon))
        
        for n in tqdm(range(2, self.nmax + 1), desc='Computing disturbing potential'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * dPnm[:, n, m]
            dTdtheta += (self.shc['a'] / r)**n * sum
        
        dTdtheta = self.shc['GM'] / r * dTdtheta # [m2/s2/rad]
        
        return dTdtheta
    
    def disturbing_potential_derivative_parallel(self, r_or_R='r') -> np.ndarray:
        '''
        Compute disturbing potential using chunking and Numba optimization.
        
        Parameters
        ----------
        r_or_R    : Either 'r' for radial distance or 'R' for reference radius
        Returns
        -------
        dTdtheta  : Disturbing potential array (m2/s2)
        '''
    
        r = self.shc['a'] * np.ones(len(self.lon)) if r_or_R == 'R' else self.r

        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a = np.array(self.shc['a'])
        GM = np.array(self.shc['GM'])
        
        dTdtheta = np.zeros(len(self.lon))
                
        n_points = len(self.lon)
        n_chunks = (n_points // self.chunk) + 1
        
        for i in range(n_chunks):
            start_idx = i * self.chunk
            end_idx = min((i + 1) * self.chunk, n_points)
            
            lon_chunk = self.lon[start_idx:end_idx]
            r_chunk = r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            T_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)  # Clear the output in Jupyter notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...') #, end='\r')
            _, dPnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                T_chunk += compute_disturbing_potential_derivative_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, dPnm_chunk, n)

            dTdtheta[start_idx:end_idx] = T_chunk
            # print('\n')
        
        dTdtheta = GM / r * dTdtheta # [m2/s2/rad]
        
        return dTdtheta
    
    
    def disturbing_potential_derivative(self, r_or_R='r', parallel: bool=True, batch_size: int = 1000) -> np.ndarray:
        '''
        Method to call either disturbing_potential_derivative_parallel() or disturbing_potential_derivative_sequential()
        with automatic batching for large arrays
        
        Parameters
        ----------
        r_or_R     : Either 'r' for radial distance or 'R' for reference radius
        parallel   : Whether to use parallel processing (forced to False for large arrays)
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        dTdtheta   : Derivative of disturbing potential [m2/s2/rad]
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('disturbing_potential_derivative', batch_size=batch_size, 
                                     r_or_R=r_or_R, parallel=parallel)
        
        # Original implementation for smaller arrays
        if parallel:
            return self.disturbing_potential_derivative_parallel(r_or_R=r_or_R)
        else:
            return self.disturbing_potential_derivative_sequential(r_or_R=r_or_R)

    def ellipsoidal_correction(self, parallel: bool = True, r_or_R='r', batch_size: int = 1000) -> np.ndarray:
        '''
        Compute the ellipsoidal correction for gravity anomalies with automatic batching for large arrays
        
        Parameters
        ----------
        parallel   : whether to use parallel processing (forced to False for large arrays)
        r_or_R     : Either 'r' for radial distance or 'R' for reference radius
        batch_size : Maximum batch size for large arrays (default: 1000)
        
        Returns
        -------
        Dg_ell     : Ellipsoidal correction [mGal], shaped as input coordinates or grid [mGal]
        
        References
        ----------
        1. Jekeli (1981): The Downward Continuation to the Earth's Surface of Truncated Spherical and Ellipsoidal Harmonic Series of the Gravity and Height Anomalies
        '''
        # Check if batching is needed
        if len(self.lon) > batch_size:
            return self._batch_process('ellipsoidal_correction', batch_size=batch_size, 
                                     parallel=parallel, r_or_R=r_or_R)
        
        # Original implementation for smaller arrays
        if not all(col in self.grav_data.columns for col in ['lon', 'lat']) or not ('elevation' in self.grav_data.columns or 'height' in self.grav_data.columns):
            raise ValueError('grav_data must contain columns: lon, lat, elevation | height')
        
        
        print('Ellipsoidal correction requires the disturbing potential and its derivative.\nComputing disturbing potential...')
        T = self.disturbing_potential(r_or_R=r_or_R, parallel=parallel)
        print('Computing the first derivative of disturbing potential with respect to colatitude...')
        dT = self.disturbing_potential_derivative(parallel=parallel, r_or_R=r_or_R)
        
        print('Computing ellipsoidal correction...')        
        # Negate dT since it is the first derivative of the potential with respect to colatitude
        dT_dphi = -dT

        ellipsoid = constants.wgs84() if self.ellipsoid.lower() == 'wgs84' else constants.grs80()
        e2 = ellipsoid['e2']
        # GM = ellipsoid['GM']
        R = ellipsoid['semi_major']

        phi = np.radians(self.grav_data['lat'].values)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        Delta_g_ell = -e2 / R * (sin_phi * cos_phi * dT_dphi - (3 * sin_phi**2 - 2) * T) * 1e5
        
        print('Ellipsoidal correction computed successfully.')
        return Delta_g_ell

    def _batch_process(self, method_name, batch_size=1000, **kwargs):
        '''
        Automatically batch process large arrays to avoid memory issues
        
        Parameters
        ----------
        method_name : Name of the method to call ('geoid', 'disturbing_potential', etc.)
        batch_size  : Maximum batch size (default: 1000)
        **kwargs    : Arguments to pass to the method
        
        Returns
        -------
        result : Concatenated results from all batches
        '''
        n_points = len(self.lon)
        
        if n_points <= batch_size:
            # No batching needed
            method = getattr(self, method_name)
            return method(**kwargs)
        
        # Batching needed
        print(f"Large array detected ({n_points} points). Processing in batches of {batch_size} to avoid memory issues...")
        
        # Force parallel=False for large arrays to prevent memory issues
        if kwargs.get('parallel', False):
            print("Forcing parallel=False for large arrays to prevent memory issues.")
            kwargs['parallel'] = False
        
        results = []
        n_batches = (n_points + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            print(f"Processing batch {i + 1}/{n_batches} (points {start_idx + 1}-{end_idx})...")
            
            # Create a temporary GGM object for this batch
            batch_ggm = GlobalGeopotentialModel(
                shc=self.shc,
                model_name=self.model,
                ellipsoid=self.ellipsoid,
                nmax=self.nmax,
                grav_data=pd.DataFrame({
                    'lon': self.lon[start_idx:end_idx],
                    'lat': self.lat[start_idx:end_idx],
                    'height': self.h[start_idx:end_idx]
                }),
                zonal_harmonics=False,  # Already done in original object
                model_dir=self.model_dir,
                chunk_size=self.chunk,
                dtm_model=self.dtm_model
            )
            
            # Call the method on this batch
            method = getattr(batch_ggm, method_name)
            batch_result = method(**kwargs)
            results.append(batch_result)
        
        # Concatenate results
        final_result = np.concatenate(results)
        print(f"Completed processing all {n_batches} batches.")
        
        return final_result

        
class GlobalGeopotentialModel2D():
    def __init__(
        self, 
        shc=None, 
        model_name=None, 
        ellipsoid='wgs84', 
        nmax=90, 
        lon=None, 
        lat=None, 
        height=None, 
        grid_spacing=1,
        zonal_harmonics=True, 
        model_dir='downloads'
    ) -> None:
        '''
        Initialize the GlobalGeopotentialModel2D class
        
        Parameters
        ----------
        shc            : spherical harmonic coefficients
        model_name     : model name
        ellipsoid      : ellipsoid
        nmax           : maximum degree
        lon            : Longitude (1D array) -- units of degree
        lat            : Latitude (1D array)  -- units of degree
        height         : Height (1D array)
        grid_spacing   : grid spacing (in degrees)
        zonal_harmonics: whether to subtract zonal harmonics
        '''
        self.shc          = shc
        self.model        = model_name
        self.ellipsoid    = ellipsoid
        self.nmax         = nmax
        self.lon          = lon
        self.lat          = lat
        self.grid_spacing = grid_spacing
        self.h            = height
        self.model_dir    = model_dir
        
        if self.model_dir is None:
            self.model_dir = 'downloads'
            
        if self.model.endswith('.gfc'):
            self.model = self.model.split('.gfc')[0]
            
        if self.shc is None and self.model is None:
            raise ValueError('Either shc or model_name must be specified')
        
        if self.shc is None:
            # self.shc = icgem.read_icgem(os.path.join(self.model_dir, self.model + '.gfc'))
            self.shc = icgem.read_icgem(str(Path(self.model_dir) / (self.model + '.gfc')))
        # Subtract even zonal harmonics
        self.shc = shtools.subtract_zonal_harmonics(self.shc, self.ellipsoid) if zonal_harmonics else self.shc
        
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
        self.n = np.arange(self.shc['nmax'] + 1)
        self.cosm = np.cos(np.arange(0, self.shc['nmax']+1)[:, np.newaxis] * self.lambda_)
        self.sinm = np.sin(np.arange(0, self.shc['nmax']+1)[:, np.newaxis] * self.lambda_)
 

    def gravity_anomaly_2D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Vectorized computations of gravity anomalies on a grid
        
        Returns
        -------
        Dg     : Gravity anomaly array (2D array)
        '''
        degree_term = self.shc['GM'] / self.shc['a'] ** 2 * (self.n - 1) * 10 ** 5 # mGal
        # Set degrees 0 and 1 to zero
        degree_term[0:2] = 0
        
        Dg = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing gravity anomalies'):
            Pnm, _ = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            Dg[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, Dg
    
    def gravity_disturbance_2D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Vectorized computations of gravity disturbances on a grid
        
        Returns
        -------
        dg     : Gravity disturbance array (2D array)
        '''
        degree_term = self.shc['GM'] / self.shc['a'] ** 2 * (self.n + 1) * 10 ** 5 # mGal
        # Set degrees 0 and 1 to zero
        degree_term[0:2] = 0
        
        dg = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing gravity disturbances'):
            Pnm, _ = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            dg[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, dg
    
    def second_radial_derivative_2D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Vectorized computations of second radial derivative on a grid
        
        Returns
        -------
        Tzz    : Second radial derivative array (2D array)
        '''
        degree_term = self.shc['GM'] / self.shc['a'] ** 3 * (self.n + 1) * (self.n + 2) * 10 ** 9 # E
        # Set degrees 0 and 1 to zero
        degree_term[0:2] = 0
        
        Tzz = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing vertical gradient'):
            Pnm, _ = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            Tzz[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, Tzz
    
    def geoid_2D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Vectorized computations of geoid on a grid
        
        Returns
        -------
        N      : Geoid array (2D array)
        '''
        degree_term = np.ones(len(self.n)) * self.shc['a']
        # Set degrees 0 and 1 to zero
        degree_term[0:2] = 0
        
        N = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing vertical gradient'):
            Pnm, _ = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            N[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, N

    def disturbing_potential_2D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Vectorized computations of anomalous potential on a grid
        
        Returns
        -------
        T      : Disturbing potential array (2D array)
        '''
        degree_term = np.ones(len(self.n)) * self.shc['GM'] / self.shc['a']
        # Set degrees 0 and 1 to zero
        degree_term[0:2] = 0
        
        T = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing gravity anomalies'):
            Pnm, _ = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            T[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, T