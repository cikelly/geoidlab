import icgem
import shtools
import constants
import gravity

import numpy as np
import pandas as pd
import coordinates as co

from legendre import ALFsGravityAnomaly
from numba import jit
from numba_progress import ProgressBar

class GlobalGeopotentialModel():
    def __init__(self, shc=None, model_name=None, ellipsoid='wgs84', nmax=300, grav_data=None, chunk_size=100, split_data=False):
        '''
        Initialize the GlobalGeopotentialModel class
        
        Parameters
        ----------
        shc       : Spherical Harmonic Coefficients -- output of icgem.read_icgem()
        ellipsoid : Reference ellipsoid ('wgs84' or 'grs80')
        nmax      : Maximum spherical harmonic degree
        grav_data : Gravity data with columns lon, lat, and elevation: lat and lon units: degrees
        '''
        self.shc       = shc
        self.model     = model_name
        self.ellipsoid = ellipsoid
        self.nmax      = nmax
        self.grav_data = grav_data
        self.chunk     = chunk_size
        self.split     = split_data
        
        if self.shc is None and self.model is None:
            raise ValueError('Either shc or model_name must be specified')
        
        if self.shc is None:
            self.shc = icgem.read_icgem(self.model)
        # Subtract even zonal harmonics
        self.shc = shtools.replace_zonal_harmonics(self.shc, self.ellipsoid)
        
        if self.grav_data is None:
            raise ValueError('Provide data with columns lon, lat, and elevation in order')
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
    
    def compute_gravity_for_chunk(self, Cnm, Snm, lon, a, GM, r, Pnm, nmax, Dg):
        '''
        Compute gravity anomaly for a chunk of data
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        GM       : Gravitational constant times the mass of the Earth
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
        2. Please ensure that you have called shtools.replace_zonal_harmonics() on shc before passing it to gravity_anomaly()
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a   = np.array(self.shc['a'])
        GM  = np.array(self.shc['GM'])
        
        Dg = np.zeros(len(self.lon))
        
        if not self.split:
            lon = np.radians(self.lon)
            Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
            Dg = self.compute_gravity_for_chunk(Cnm, Snm, lon, a, GM, self.r, Pnm, self.nmax, Dg)
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
                
                Dg_chunk = self.compute_gravity_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, GM, r_chunk, Pnm_chunk, self.nmax, Dg_chunk)
                Dg[start_idx:end_idx] = Dg_chunk
                print('\n')
        Dg = GM / self.r ** 2 * Dg * 10**5 # mGal
        
        return pd.Series(Dg)
            
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
    
    def compute_disturbance_for_chunk(self, Cnm, Snm, lon, a, GM, r, Pnm, nmax, dg):
        '''
        Compute disturbance for a chunk of data
        
        Parameters
        ----------
        Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
        lon      : Longitude (1D arrays)
        a        : Reference radius 
        GM       : Gravitational constant times the mass of the Earth
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
        2. Please ensure that you have called shtools.replace_zonal_harmonics() on shc before passing it to gravity_disturbance()
        '''
        Cnm = np.array(self.shc['Cnm'])
        Snm = np.array(self.shc['Snm'])
        a   = np.array(self.shc['a'])
        GM  = np.array(self.shc['GM'])
        
        dg = np.zeros(len(self.lon))
        
        if not self.split:
            lon = np.radians(self.lon)
            Pnm = ALFsGravityAnomaly(vartheta=self.vartheta, nmax=self.nmax, ellipsoid=self.ellipsoid)
            dg = self.compute_disturbance_for_chunk(Cnm, Snm, lon, a, GM, self.r, Pnm, self.nmax, dg)
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
                
                dg_chunk = self.compute_disturbance_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, GM, r_chunk, Pnm_chunk, self.nmax, dg_chunk)
                dg[start_idx:end_idx] = dg_chunk
                print('\n')
        dg = GM / self.r ** 2 * dg * 10**5 # mGal
        
        return pd.Series(dg)

    def zero_degree_term(self, geoid=None, GM=None):
        '''
        Add zero-degree term to the GGM geoid
        
        Parameters
        ----------
        geoid      : Geoid model (output of ggm_tools.reference_geoid())
        GM         : Gravity constant of the GGM
        
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
        
        W0  = constants.earth('W0')
        R   = constants.earth('R')
        
        gamma_0 = gravity.normal_gravity(phi=self.lat, ellipsoid=self.ellipsoid)
        
        N = geoid + ( (GM - GM0) / R - (W0 - U0) ) / gamma_0 
        
        return N