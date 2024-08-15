############################################################
# Utilities for synthesizing gravity functionals           #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import icgem
import shtools
import constants
import gravity
import os

import numpy as np
import pandas as pd
import coordinates as co

from legendre import ALFsGravityAnomaly, ALF
from numba import jit
from numba_progress import ProgressBar
from tqdm import tqdm

class GlobalGeopotentialModel():
    def __init__(
        self, shc=None, model_name=None, 
        ellipsoid='wgs84', nmax=90, 
        grav_data=None, chunk_size=None, 
        zonal_harmonics=True,
        model_dir='downloads'
    ):
        '''
        Initialize the GlobalGeopotentialModel class for potential modeling
        
        Parameters
        ----------
        shc             : Spherical Harmonic Coefficients -- output of icgem.read_icgem()
        model_name      : Name of gravity model without extension
        ellipsoid       : Reference ellipsoid ('wgs84' or 'grs80')
        nmax            : Maximum spherical harmonic degree
        grav_data       : Gravity data with columns lon, lat, and elevation: lat and lon units: degrees
        chunk_size      : If provided, data is processed in chunk of length chunk_size
        zonal_harmonics : Whether to subtract even zonal harmonics or not
        model_dir       : Directory where model file is stored
        '''
        self.shc       = shc
        self.model     = model_name
        self.ellipsoid = ellipsoid
        self.nmax      = nmax
        self.grav_data = grav_data
        self.chunk     = chunk_size
        self.model_dir = model_dir
        
        if self.chunk is not None:
            try:
                self.chunk = int(self.chunk)
            except ValueError:
                print('Invalid chunk_size. Setting chunk_size to 100.')
                self.chunk = 100
        
        if self.chunk is not None and self.chunk <= 0:
            print('Invalid chunk_size. Setting chunk_size to 100.')
            self.chunk = 100

        if self.model.endswith('.gfc'):
            self.model = self.model.split('.gfc')[0]
            
        if self.shc is None and self.model is None:
            raise ValueError('Either shc or model_name must be specified')
        
        if self.shc is None:
            self.shc = icgem.read_icgem(os.path.join(self.model_dir, self.model + '.gfc'))
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
                    # raise ValueError('Provide data with columns lon, lat, and elevation in order.')
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
        # if np.all(self.h == 0):
        #     raise ValueError('All elevations are 0. Use GlobalGeopotentialModel2D instead if you have gridded data')
        self.r, self.vartheta, _ = co.geodetic2spherical(phi=self.lat, lambd=self.lon, height=self.h, ellipsoid=self.ellipsoid)
        
        # Set self.r = self.shc['a'] if all self.h = 0 (Doesn't really matter)
        if np.all(self.h == 0):
            print('Setting r = R ...')
            self.r = self.shc['a']
            
            
    @staticmethod
    @jit(nopython=True)
    def compute_gravity_chunk(Cnm, Snm, lon, a, r, Pnm, n, Dg):
        '''
        Compute a chunk of gravity anomaly for a specific degree using Numba optimization
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        n        : Specific degree
        Dg       : Gravity anomaly array to update
        
        Returns
        -------
        Dg       : Updated gravity anomaly array for degree n
        '''
        sum = np.zeros(len(lon))
        for m in range(n + 1):
            sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
        Dg += (n - 1) * (a / r) ** n * sum
        
        return Dg
    
    def compute_gravity_for_chunk(self, Cnm, Snm, lon, a, r, Pnm, nmax, Dg):
        '''
        Compute gravity anomaly for a chunk of data
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        nmax     : Maximum spherical harmonic degree of expansion
        Dg       : Gravity anomaly array to update

        Returns
        -------
        Updated Dg array with computed values for all degrees
        '''
        with ProgressBar(total=nmax - 1, desc='Calculating Gravity Anomalies') as pbar:
            for n in range(2, nmax + 1):
                Dg = self.compute_gravity_chunk(Cnm, Snm, lon, a, r, Pnm, n, Dg)
                pbar.update(1)
        
        return Dg
    
    def gravity_anomaly(self):
        '''
        Wrapper function to handle data and call the Numba-optimized function
        
        Returns
        -------
        Dg       : Gravity anomaly array (mGal)
        
        Notes
        -----
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        2. Please ensure that you have called shtools.subtract_zonal_harmonics() on shc before passing it to gravity_anomaly()
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a   = np.array(self.shc['a'])
        GM  = np.array(self.shc['GM'])
        
        Dg = np.zeros(len(self.lon))
        
        if self.chunk is None or self.chunk >= len(self.lon):
            lon = np.radians(self.lon)
            Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
            Dg = self.compute_gravity_for_chunk(Cnm, Snm, lon, a, self.r, Pnm, self.nmax, Dg)
        else:
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
                
                print(f'Processing chunk {i + 1} of {n_chunks}...')
                Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid)
                
                Dg_chunk = self.compute_gravity_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, r_chunk, Pnm_chunk, self.nmax, Dg_chunk)
                Dg[start_idx:end_idx] = Dg_chunk
                print('\n')
        Dg = GM / self.r ** 2 * Dg * 10**5 # mGal
        
        return Dg
            
    @staticmethod
    @jit(nopython=True)
    def compute_disturbance_chunk(Cnm, Snm, lon, a, r, Pnm, n, dg):
        '''
        Compute a chunk of gravity disturbances for a specific degree using Numba optimization
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        n        : Specific degree
        dg       : Gravity disturbance array to update
        
        Returns
        -------
        dg       : Updated disturbance disturbance array for degree n
        '''
        sum = np.zeros(len(lon))
        for m in range(n + 1):
            sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
        dg += (n + 1) * (a / r) ** n * sum
        
        return dg
    
    def compute_disturbance_for_chunk(self, Cnm, Snm, lon, a, r, Pnm, nmax, dg):
        '''
        Compute disturbance for a chunk of data
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        nmax     : Maximum spherical harmonic degree of expansion
        dg       : Disturbance array to update

        Returns
        -------
        Updated dg array with computed values for all degrees
        '''
        with ProgressBar(total=nmax - 1, desc='Calculating Gravity Disturbance') as pbar:
            for n in range(2, nmax + 1):
                dg = self.compute_disturbance_chunk(Cnm, Snm, lon, a, r, Pnm, n, dg)
                pbar.update(1)
        
        return dg
    
    def gravity_disturbance(self):
        '''
        Wrapper function to handle data and call the Numba-optimized function
        
        Returns
        -------
        dg       : Gravity Disturbance array (mGal)
        
        Notes
        -----
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        2. Please ensure that you have called shtools.subtract_zonal_harmonics() on shc before passing it to gravity_disturbance()
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a   = np.array(self.shc['a'])
        GM  = np.array(self.shc['GM'])
        
        dg = np.zeros(len(self.lon))
        
        if self.chunk is None or self.chunk >= len(self.lon):
            lon = np.radians(self.lon)
            Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
            dg = self.compute_disturbance_for_chunk(Cnm, Snm, lon, a, self.r, Pnm, self.nmax, dg)
        else:
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
                
                print(f'Processing chunk {i + 1} of {n_chunks}...')
                Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid)
                
                dg_chunk = self.compute_disturbance_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, r_chunk, Pnm_chunk, self.nmax, dg_chunk)
                dg[start_idx:end_idx] = dg_chunk
                print('\n')
        dg = GM / self.r ** 2 * dg * 10**5 # mGal
        
        return dg

    def zero_degree_term(self, geoid=None, GM=None, zeta_or_geoid='geoid'):
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
            gamma_0 = gravity.normal_gravity_above_ellipsoid(
                phi=self.lat, h=self.h, ellipsoid=self.ellipsoid
            ) # This is actually gamma_Q
            gamma_0 = gamma_0 / 1e5 # mgal to m/s2
        else:
            gamma_0 = gravity.normal_gravity(phi=self.lat, ellipsoid=self.ellipsoid)
        
        N = geoid + ( (GM - GM0) / R - (W0 - U0) ) / gamma_0 
        
        return N
    
        
    @staticmethod
    @jit(nopython=True)
    def compute_radial_chunk(Cnm, Snm, lon, a, r, Pnm, n, Tzz):
        '''
        Compute a chunk of second radial derivative of the disturbing potential 
        for a specific degree using Numba optimization -- vertical gravity gradient
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        n        : Specific degree
        Tzz      : Radial derivative array to update (unit: 1E = 10^{−9}s^{−2})
        
        Returns
        -------
        Tzz      : Updated second radial derivative array for degree n
        '''
        sum = np.zeros(len(lon))
        for m in range(n + 1):
            sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
        Tzz += (n + 1) * (n + 2) * (a / r) ** n * sum
        
        return Tzz
    
    def compute_radial_for_chunk(self, Cnm, Snm, lon, a, r, Pnm, nmax, Tzz):
        '''
        Compute radial for a chunk of data
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        nmax     : Maximum spherical harmonic degree of expansion
        Tzz       : Radial array to update (Eötvös (E): 10^{−9}s^{−2})

        Returns
        -------
        Updated Tzz array with computed values for all degrees
        '''
        with ProgressBar(total=nmax - 1, desc='Calculating Radial Derivative') as pbar:
            for n in range(2, nmax + 1):
                Tzz = self.compute_radial_chunk(Cnm, Snm, lon, a, r, Pnm, n, Tzz)
                pbar.update(1)
        
        return Tzz
    
    def second_radial_derivative(self):
        '''
        Wrapper function to handle data and call the Numba-optimized function
        
        Returns
        -------
        Tzz       : Second radial derivative array (Eötvös (E): 10^{−9}s^{−2})
        
        Notes
        -----
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.39, p.298
        2. Please ensure that you have called shtools.subtract_zonal_harmonics() on shc before passing it to second_radial()
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a   = np.array(self.shc['a'])
        GM  = np.array(self.shc['GM'])
        
        Tzz = np.zeros(len(self.lon))
        
        if self.chunk is None or self.chunk >= len(self.lon):
            lon = np.radians(self.lon)
            Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
            Tzz = self.compute_radial_for_chunk(Cnm, Snm, lon, a, self.r, Pnm, self.nmax, Tzz)
        else:
            n_points = len(self.lon)
            n_chunks = (n_points // self.chunk) + 1
            print(f'Data will be processed in {n_chunks} chunks...\n')

            for i in range(n_chunks):
                start_idx = i * self.chunk
                end_idx = min((i + 1) * self.chunk, n_points)
                
                lon_chunk = self.lon[start_idx:end_idx]
                r_chunk = self.r[start_idx:end_idx]
                vartheta_chunk = self.vartheta[start_idx:end_idx]
                Tzz_chunk = np.zeros(len(lon_chunk))
                
                print(f'Processing chunk {i + 1} of {n_chunks}...')
                Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid)
                
                Tzz_chunk = self.compute_radial_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, r_chunk, Pnm_chunk, self.nmax, Tzz_chunk)
                Tzz[start_idx:end_idx] = Tzz_chunk
                print('\n')
        Tzz = GM / self.r ** 3 * Tzz * 10 ** 9 # E = Eötvös
        
        return Tzz
    
    def read_file(self):
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
    
    @staticmethod
    @jit(nopython=True)
    def compute_disturbing_potential_chunk(Cnm, Snm, lon, a, r, Pnm, n, T):
        '''
        Compute a chunk of anomalous potential for a specific degree using Numba optimization
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        n        : Specific degree
        T        : Disturbing potential array to update
        
        Returns
        -------
        T        : Updated anomalous potential array for degree n
        '''
        sum = np.zeros(len(lon))
        for m in range(n + 1):
            sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
        T += (a / r) ** n * sum
        
        return T
    
    def compute_disturbing_potential_for_chunk(self, Cnm, Snm, lon, a, r, Pnm, nmax, T):
        '''
        Compute gravity anomaly for a chunk of data
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        r        : Radial distance (1D array)
        Pnm      : Associated Legendre functions (3D array)
        nmax     : Maximum spherical harmonic degree of expansion
        T        : Anomalous potential array to update

        Returns
        -------
        Updated T array with computed values for all degrees
        '''
        with ProgressBar(total=nmax - 1, desc='Calculating Disturbing Potential') as pbar:
            for n in range(2, nmax + 1):
                T = self.compute_disturbing_potential_chunk(Cnm, Snm, lon, a, r, Pnm, n, T)
                pbar.update(1)
        
        return T
    
    def disturbing_potential(self):
        '''
        Wrapper function to handle data and call the Numba-optimized function
        
        Returns
        -------
        T       : Disturbing potential array (m2/s2)
        
        Notes
        -----
        1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
        2. Please ensure that you have called shtools.subtract_zonal_harmonics() on shc before passing it to disturbing_potential()
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a   = np.array(self.shc['a'])
        GM  = np.array(self.shc['GM'])
        
        T = np.zeros(len(self.lon))
        
        if self.chunk is None or self.chunk >= len(self.lon):
            lon = np.radians(self.lon)
            Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
            T = self.compute_disturbing_potential_for_chunk(Cnm, Snm, lon, a, self.r, Pnm, self.nmax, T)
        else:
            n_points = len(self.lon)
            n_chunks = (n_points // self.chunk) + 1
            print(f'Data will be processed in {n_chunks} chunks...\n')

            for i in range(n_chunks):
                start_idx = i * self.chunk
                end_idx = min((i + 1) * self.chunk, n_points)
                
                lon_chunk = self.lon[start_idx:end_idx]
                r_chunk = self.r[start_idx:end_idx]
                vartheta_chunk = self.vartheta[start_idx:end_idx]
                T_chunk = np.zeros(len(lon_chunk))
                
                print(f'Processing chunk {i + 1} of {n_chunks}...')
                Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=self.nmax, ellipsoid=self.ellipsoid)
                
                T_chunk = self.compute_disturbing_potential_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, r_chunk, Pnm_chunk, self.nmax, T_chunk)
                T[start_idx:end_idx] = T_chunk
                print('\n')
        T = GM / self.r * T # m2/s2
        # T = T
        
        return T

    
    def height_anomaly(self, T=None):
        '''
        Height anomaly based on Bruns' method
        
        Parameters
        ----------
        T         : Disturbing potential array (m2/s2)
        
        Returns
        -------
        zeta    : Height anomaly (m)
        
        Notes
        -----
        1. Torge, Muller, & Pail (2023): Geodesy, Eq. 6.9, p.288
        '''
        print('Using Bruns\' method to calculate height anomaly...\n')
        
        T = self.disturbing_potential() if T is None else T
        gammaQ = gravity.normal_gravity_above_ellipsoid(phi=self.lat, h=self.h, ellipsoid=self.ellipsoid)
        gammaQ = gammaQ * 1e-5 # mgal to m/s2
        zeta = T/gammaQ
        # zeta = (T * self.shc['GM'] / self.r) / gammaQ
        
        # Zero-degree term
        zeta = self.zero_degree_term(geoid=zeta, zeta_or_geoid='zeta')
        
        return zeta #+ zeta_0
    
    def geoid(self, T=None):
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
        print('Using Bruns\' method to calculate geoid height...\n')
        
        T = self.disturbing_potential() if T is None else T
        gamma0 = gravity.normal_gravity(phi=self.lat, ellipsoid=self.ellipsoid)

        N = T / gamma0
        
        # Zero-degree term
        N = self.zero_degree_term(geoid=N, zeta_or_geoid='geoid')
        
        return N
    
class GlobalGeopotentialModel2D():
    def __init__(
        self, shc=None, model_name=None, 
        ellipsoid='wgs84', nmax=90, 
        lon=None, lat=None, 
        height=None, grid_spacing=1,
        zonal_harmonics=True, model_dir='downloads'
    ):
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
        
        if self.model.endswith('.gfc'):
            self.model = self.model.split('.gfc')[0]
            
        if self.shc is None and self.model is None:
            raise ValueError('Either shc or model_name must be specified')
        
        if self.shc is None:
            self.shc = icgem.read_icgem(os.path.join(self.model_dir, self.model + '.gfc'))
        # Subtract even zonal harmonics
        self.shc = shtools.subtract_zonal_harmonics(self.shc, self.ellipsoid) if zonal_harmonics else self.shc
        
        if self.lon is None and self.lat is None:
            print('No grid coordinates provided. Computing over the entire globe...\n')
            self.lambda_ = np.radians(np.arange(-180, 180+self.grid_spacing, self.grid_spacing))
            self.theta   = np.radians(np.arange(0, 180+self.grid_spacing, self.grid_spacing)) # co-latitude
            self.lon     = np.arange(-180, 180+self.grid_spacing, self.grid_spacing)
            self.lat     = np.arange(-90, 90+self.grid_spacing, self.grid_spacing)
            # print(self.lon.shape, self.lat.shape)
            # self.r, _ ,_ = co.geodetic2spherical(
            #     phi=self.lat, lambd=self.lon, 
            #     height=len(self.lon), ellipsoid=self.ellipsoid
            # )
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
 

    def gravity_anomaly_2D(self):
        '''
        Vectorized computations of gravity anomalies on a grid
        
        Returns
        -------
        Dg     : Gravity anomaly array (2D array)
        '''
        degree_term = self.shc['GM'] / self.shc['a'] ** 2 * (self.n - 1) * 10 ** 5 # mGal
        # degree_term = self.shc['GM'] / self.r ** 2 * (self.n - 1) *  self.shc['a']/self.r ** self.n * 10 ** 5 # mGal
        # Set degrees 0 and 1 to zero
        degree_term[0:2] = 0
        
        Dg = np.zeros((len(self.theta), len(self.lambda_)))
        
        for i, theta_ in tqdm(enumerate(self.theta), total=len(self.theta), desc='Computing gravity anomalies'):
            Pnm = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            Dg[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, Dg
    
    def gravity_disturbance_2D(self):
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
            Pnm = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            dg[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, dg
    
    def second_radial_derivative_2D(self):
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
            Pnm = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            Tzz[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, Tzz
    
    def geoid_2D(self):
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
            Pnm = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            N[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, N

    def disturbing_potential_2D(self):
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
            Pnm = ALF(vartheta=theta_, nmax=self.shc['nmax'], ellipsoid=self.ellipsoid)
            T[i,:] = degree_term @ ((self.shc['Cnm'] * Pnm) @ self.cosm + (self.shc['Snm'] * Pnm) @ self.sinm)
            
        return self.Lon, self.Lat, T