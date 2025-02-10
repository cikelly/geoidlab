############################################################
# Utilities for synthesizing gravity functionals           #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from pathlib import Path

from . import (
    icgem,
    constants,
    gravity,
    coordinates as co,
    shtools
)

from easy_pygeoid.legendre import ALFsGravityAnomaly, ALF
from .utils.parallel_utils import (
    compute_gravity_chunk, 
    compute_disturbance_chunk,
    compute_disturbing_potential_chunk,
    compute_second_radial_chunk,
    compute_separation_chunk
)
from numba import jit
from numba_progress import ProgressBar
from tqdm import tqdm
from joblib import Parallel, delayed
from IPython.display import clear_output

import numpy as np
import pandas as pd
import gc

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
        chunk_size=None
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
        '''
        self.shc       = shc
        self.model     = model_name
        self.ellipsoid = ellipsoid
        self.nmax      = nmax
        self.grav_data = grav_data
        self.model_dir = model_dir
        self.chunk     = chunk_size
        
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
        Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
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
            
            clear_output(wait=True)  # Clear the output in Jupyter notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...', end='\r')
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                Dg_chunk += compute_gravity_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)
            
            Dg[start_idx:end_idx] = Dg_chunk
            print('\n')
        
        return GM / self.r ** 2 * Dg * 10**5  # Convert to mGal
    
    def gravity_anomaly(self, parallel: bool=True) -> np.ndarray:
        '''
        Method to call either gravity_anomaly_parallel() or gravity_anomaly_sequential()
        
        Returns
        -------
        Dg : Gravity anomaly array (mGal)
        '''
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
        
        Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
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
            print(f'Processing chunk {i + 1} of {n_chunks}...', end='\r')
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                dg_chunk += compute_disturbance_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)

            dg[start_idx:end_idx] = dg_chunk
            print('\n')
        
        return GM / self.r ** 2 * dg * 10**5  # [mGal]
    
    def gravity_disturbance(self, parallel: bool=True) -> np.ndarray:
        '''
        Method to call either gravity_disturbance_parallel() or gravity_disturbance_sequential()
        
        Returns
        -------
        dg : Gravity disturbance array (mGal)
        '''
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
        if r_or_R == 'R':
            self.r = self.shc['a'] * np.ones(len(self.lon))
            self.R = self.shc['a'] * np.ones(len(self.lon))
            
        Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        T   = np.zeros(len(self.lon))
        
        for n in tqdm(range(2, self.nmax + 1), desc='Computing disturbing potential'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            T += (self.shc['a'] / self.r)**n * sum
        if r_or_R == 'r':
            T = self.shc['GM'] / self.r * T # [m2/s2]
        else:
            T = self.shc['GM'] / self.R * T # [m2/s2]        
        
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
        if r_or_R == 'R':
            self.r = self.shc['a'] * np.ones(len(self.lon))
            self.R = self.shc['a'] * np.ones(len(self.lon))
            
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
            r_chunk = self.r[start_idx:end_idx]
            vartheta_chunk = self.vartheta[start_idx:end_idx]
            T_chunk = np.zeros(len(lon_chunk))
            
            clear_output(wait=True)  # Clear the output in Jupyter notebook
            print(f'Processing chunk {i + 1} of {n_chunks}...', end='\r')
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                T_chunk += compute_disturbing_potential_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)

            T[start_idx:end_idx] = T_chunk
            print('\n')
        
        if r_or_R == 'r':
            T = GM / self.r * T # [m2/s2]
        else:
            T = GM / self.R * T # [m2/s2]
        
        return T
    
    
    def disturbing_potential(self, r_or_R='r', parallel: bool=True) -> np.ndarray:
        '''
        Method to call either disturbing_potential_parallel() or disturbing_potential_sequential()
        
        Parameters
        ----------
        r_or_R    : Either 'r' for radial distance or 'R' for reference radius
        Returns
        -------
        T         : Disturbing potential array (m2/s2)
        '''
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
        Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
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
            print(f'Processing chunk {i + 1} of {n_chunks}...', end='\r')
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                Tzz_chunk += compute_second_radial_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)

            Tzz[start_idx:end_idx] = Tzz_chunk
            print('\n')
        
        
        return GM / self.r ** 3 * Tzz * 10**9 # [E = Eötvös]
    
    def second_radial_derivative(self, parallel: bool=True) -> np.ndarray:
        '''
        Method to call either second_radial_derivative_parallel() or second_radial_derivative_sequential()
        
        Returns
        -------
        Tzz    : Second radial derivative (Eötvös (E): 10^{−9}s^{−2})
        '''
        if parallel:
            return self.second_radial_derivative_parallel()
        else:
            return self.second_radial_derivative_sequential()
        
    def height_anomaly(
        self, 
        T=None, 
        tolerance=5e-3, 
        max_iter=5,
        parallel: bool=True
    ) -> np.ndarray:
        '''
        Height anomaly based on Bruns' method
        
        Parameters
        ----------
        T         : Disturbing potential array (m2/s2)
        tolerance : Tolerance for refining height anomaly
        max_iter  : Maximum number of iterations
        parallel  : True or False for parallel processing
        
        Returns
        -------
        zeta      : Height anomaly (m)
        
        Notes
        -----
        1. Torge, Muller, & Pail (2023): Geodesy, Eq. 6.9, p.288
        2. We need to iterate to refine zeta because we start estimating zeta using `H` not `hn`
            Steps:
                1. Estimate hn (normal height): hn = self.h - zeta
                2. Use hn to calculate gammaQ
                3. Use gammaQ to calculate updated zeta (zeta1)
                4. Repeat until convergence.
        '''
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
    
    def geoid(self, T=None) -> np.ndarray:
        '''
        Geoid heights based on Bruns' method
        
        Parameters
        ----------
        T         : Disturbing potential array (m2/s2)
        
        Returns
        -------
        zeta    : Geoid height (m)
        
        Notes
        -----
        1. Torge, Muller, & Pail (2023): Geodesy, Eq. 6.8, p.288
        '''
        print('Using Bruns\' method with zero-degree correction to calculate geoid height...\n')
        
        T = self.disturbing_potential(r_or_R='R') if T is None else T
        gamma0 = gravity.normal_gravity_somigliana(phi=self.lat, ellipsoid=self.ellipsoid)

        N = T / gamma0
        
        # Zero-degree term
        N = self.zero_degree_term(geoid=N, zeta_or_geoid='geoid')
        
        return N

    def zero_degree_term(self, geoid=None, GM=None, zeta_or_geoid='geoid') -> np.ndarray:
        '''
        Add zero-degree term to the GGM geoid
        
        Parameters
        ----------
        geoid         : Geoid model (output of ggm_tools.reference_geoid())
        GM            : Gravity constant of the GGM
        zeta_or_geoid : 'zeta' or 'geoid'
        Returns
        -------
        N          : Geoid corrected for zero-degree term
        
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
        
        W0  = constants.earth()['W0']
        # R   = constants.earth()['radius']
        R = self.r
        
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
        H : Geoid-quasi geoid separation values.
        '''
        Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
        H = np.zeros(len(self.lon))

        for n in tqdm(range(2, self.nmax + 1), desc='Computing separation'):
            sum = np.zeros(len(self.lon))
            for m in range(n + 1):
                mlambda = m * self.lambda_
                sum += (
                    self.shc['Cnm'][n, m] * np.cos(mlambda) + 
                    self.shc['Snm'][n, m] * np.sin(mlambda)
                ) * Pnm[:, n, m]
            H += (self.shc['a'] / self.r) ** n * sum
        
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
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid, show_progress=False)
            lon_rad_chunk = np.radians(lon_chunk)
            
            for n in range(2, self.nmax + 1):
                H_chunk += compute_separation_chunk(Cnm, Snm, lon_rad_chunk, a, r_chunk, Pnm_chunk, n)
            
            H[start_idx:end_idx] = H_chunk
            print('\n')
        
        return H
    
    def separation(self, parallel: bool=True) -> np.ndarray:
        '''
        Method to call either separation_parallel() or separation_sequential().
        
        Returns
        -------
        H : Geoid-quasi geoid separation values.
        '''
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