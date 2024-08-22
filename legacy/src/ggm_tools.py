############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from easy_pygeoid.legendre import ALF, ALFsGravityAnomaly
from easy_pygeoid.shtools import replace_zonal_harmonics
from numba import jit

import easy_pygeoid.coordinates as co
import numpy as np
import pandas as pd
from numba_progress import ProgressBar

from easy_pygeoid import gravity
from easy_pygeoid import tide
from easy_pygeoid import icgem
from easy_pygeoid import constants



def height_anomaly(
    lon, lat, height=0, nmax=300,
    shc:dict=None, model_name=None, ellipsoid='wgs84', 
    tide_sys='tide_free', downloads_dir='downloads',
):
    '''
    Compute height anomaly above the reference ellipsoid
    
    Parameters
    ----------
    lon        : geodetic/geographic longitude of point(s) of interest (degrees)
    lat        : geodetic/geographic latitude of point(s) of interest  (degrees)
    nmax       : maximum spherical harmonic degree of expansion
    shc        : the output of icgem.read_icgem()
    model_name : the name of the GGM (global gravity model) for synthesis
    
    Returns
    -------
    Zeta
    
    Notes
    -----
    1. Zeta is the height anomaly above the reference ellipsoid.
    2. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.33, p.296
    '''
    if shc is None and model_name is None:
        raise ValueError('Either shc or model_name must be specified')
    
    if shc is None and model_name is not None:
        shc = icgem.read_icgem(model_name)
    
    
    gamma   = gravity.normal_gravity(phi=lat, ellipsoid=ellipsoid)
    gamma_Q = gravity.normal_gravity_above_ellipsoid(phi=lat, h=height, ellipsoid=ellipsoid)
    # r_phi = gravity.ellipsoid_radius(lat, ellipsoid=ellipsoid)
    
    # Reference ellipsoid: remove the even zonal harmonics from sine and cosine cofficients.
    shc = replace_zonal_harmonics(shc, ellipsoid=ellipsoid)
    
    # ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    # GMe = ref_ellipsoid['GM']
    # a_e = ref_ellipsoid['semi_major']
    
    # zonal_harmonics = ['C20', 'C40', 'C60', 'C80', 'C100']
    
    # for n, Cn0 in zip([2, 4, 6, 8, 10], zonal_harmonics):
    #     shc['Cnm'][n,0] = shc['Cnm'][n,0] - ( shc['GM']/GMe) * (shc['a']/a_e )**2 * ref_ellipsoid[Cn0]
    
    # Fully normalized associated Legendre functions
    r, vartheta, _ = co.geodetic2spherical(phi=lat, lambd=lon, height=height, ellipsoid=ellipsoid)
    Pnm = ALF(phi=vartheta, nmax=nmax, ellipsoid=ellipsoid) 

    
    # Height anomaly
    
    return

def reference_geoid(
    lon, lat, height=0, nmax=300,
    shc:dict=None, model_name=None, ellipsoid='wgs84', 
    tide_sys='tide_free', downloads_dir='downloads',
):
    '''
    Compute the reference geoid from a global geopotential model
    
    Parameters
    ----------
    lon        : geodetic/geographic longitude of point(s) of interest (degrees)
    lat        : geodetic/geographic latitude of point(s) of interest  (degrees)
    nmax       : maximum spherical harmonic degree of expansion
    shc        : the output of icgem.read_icgem()
    model_name : the name of the GGM (global gravity model) for synthesis
    
    Returns
    -------
    N
    
    Notes
    -----
    1. Zeta is the height anomaly above the reference ellipsoid.
    2. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.33, p.296
    '''
    if shc is None and model_name is None:
        raise ValueError('Either shc or model_name must be specified')
    
    if shc is None and model_name is not None:
        shc = icgem.read_icgem(model_name)
    
    
    gamma   = gravity.normal_gravity(phi=lat, ellipsoid=ellipsoid)
    # gamma_Q = gravity.normal_gravity_above_ellipsoid(phi=lat, h=height, ellipsoid=ellipsoid)
    r_phi = gravity.ellipsoid_radius(lat, ellipsoid=ellipsoid)
    
    # Reference ellipsoid: remove the even zonal harmonics from sine and cosine cofficients.
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    GMe = ref_ellipsoid['GM']
    a_e = ref_ellipsoid['semi_major']
    
    zonal_harmonics = ['C20', 'C40', 'C60', 'C80', 'C100']
    
    for n, Cn0 in zip([2, 4, 6, 8, 10], zonal_harmonics):
        shc['Cnm'][n,0] = shc['Cnm'][n,0] - ( shc['GM']/GMe) * (shc['a']/a_e )**2 * ref_ellipsoid[Cn0]
    
    # Fully normalized associated Legendre functions
    Pnm = ALF(phi=r_phi, nmax=nmax, ellipsoid=ellipsoid)  
    
    return

def zero_degree_term(lat, shc=None, GM=None, geoid=None, ellipsoid='wgs84'):
    '''
    Add zero-degree term to the GGM geoid
    
    Parameters
    ----------
    lat        : Geodetic latitude of point(s) (degrees)
    shc        : Spherical Harmonic Coefficients (output of icgem.read_icgem())
    GM         : Gravity constant of the GGM
    geoid      : Geoid model (output of ggm_tools.reference_geoid())
    ellipsoid  : Reference ellipsoid (wgs84 or grs80)
    
    Returns
    -------
    N          : Geoid corrected for zero-degree term
    
    Reference
    ---------
    Hofmann-Wellenhof & Moritz (2006): Physical Geodesy, Eq. 2–356, p. 113
    '''
    if geoid is None:
        raise ValueError('Please provide geoid')
    
    if shc is None and GM is None:
        raise ValueError('Please provide shc (output of icgem.read_icgem()) or GM from GGM')
    
    if shc is not None:
        GM = shc['GM']
    
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    GMe = ref_ellipsoid['GM']
    U0  = ref_ellipsoid['U0'] # Potential of ellipsoid (m2/s2)
    
    W0  = constants.earth('W0')
    R   = constants.earth('R')
    
    gamma   = gravity.normal_gravity(phi=lat, ellipsoid=ellipsoid)
    
    N = geoid + ( (GM - GMe) / R - (W0 - U0) ) / gamma   
    
    return N


@jit(nopython=True)
def compute_gravity_chunk(Cnm, Snm, lon, a, GM, r, Pnm, n, Dg):
    '''
    Compute a chunk of gravity anomaly for a specific degree n using Numba for optimization

    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D arrays)
    a        : Reference radius 
    GM       : Gravitational constant times the mass of the Earth
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    Dg       : Gravity anomaly array to update

    Returns
    -------
    Updated Dg array with computed values for degree n
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    Dg += (n - 1) * (a / r) ** n * sum

    return Dg

def compute_gravity_for_chunk(Cnm, Snm, lon, a, GM, r, Pnm, nmax, Dg):
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
    with ProgressBar(total=nmax - 1, desc='Calculating Gravity Anomalies for chunk') as pbar:
        for n in range(2, nmax + 1):
            Dg = compute_gravity_chunk(Cnm, Snm, lon, a, GM, r, Pnm, n, Dg)
            pbar.update(1)
    
    return Dg

def gravity_anomaly(shc, grav_data=None, ellipsoid='wgs84', nmax=300, chunk_size=100, split_data=False):
    '''
    Wrapper function to handle data and call the Numba-optimized function
    
    Parameters
    ----------
    shc       : Spherical Harmonic Coefficients (output of icgem.read_icgem() and shtools.replace_zonal_harmonics())
    grav_data : Gravity data with columns lon, lat, and elevation: lat and lon units: degrees
    ellipsoid : Reference ellipsoid
    nmax      : Maximum spherical harmonic degree of expansion
    chunk_size: Size of the chunk of data to process at once
    split_data: Whether to split the data into chunks
    
    Returns
    -------
    Dg        : Gravity anomaly (mGal)
    
    Notes
    -----
    1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
    2. Please ensure that you have called shtools.replace_zonal_harmonics() on shc before passing it to gravity_anomaly()
    '''        
    if grav_data is None:
        raise ValueError('Provide data with columns lon, lat, and elevation in order')
    else:
        if isinstance(grav_data, np.ndarray):
            lon = grav_data[:, 0]
            lat = grav_data[:, 1]
            try:
                h = grav_data[:, 2]
            except IndexError:
                print('Looks like there is no elevation column. Setting elevation to 0')
                h = np.zeros(len(lat))
        elif isinstance(grav_data, pd.DataFrame):
            lon_column = [col for col in grav_data.columns if pd.Series(col).str.contains('lon', case=False).any()][0]
            lat_column = [col for col in grav_data.columns if pd.Series(col).str.contains('lat', case=False).any()][0]
            lon = grav_data[lon_column].values
            lat = grav_data[lat_column].values
            try:
                elev_column = [col for col in grav_data.columns if pd.Series(col).str.contains(r'elev|height', case=False).any()][0]
                h = grav_data[elev_column].values
            except IndexError:
                print('Looks like there is no elevation column. Setting elevation to 0')
                h = np.zeros(len(lat))
                
    r, vartheta, _ = co.geodetic2spherical(phi=lat, lambd=lon, height=h, ellipsoid=ellipsoid)
    
    Cnm = np.array(shc['Cnm'])
    Snm = np.array(shc['Snm'])
    a = shc['a']
    Gm = shc['GM']
    
    Dg = np.zeros(len(lon))

    if not split_data:
        lon = np.radians(lon)
        Pnm = ALFsGravityAnomaly(vartheta=vartheta, nmax=nmax, ellipsoid=ellipsoid)
        Dg = compute_gravity_for_chunk(Cnm, Snm, lon, a, Gm, r, Pnm, nmax, Dg)
    else:
        n_points = len(lon)
        n_chunks = (n_points // chunk_size) + 1
        print(f'Data will be processed in {n_chunks} chunks...\n')

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_points)

            lon_chunk = lon[start_idx:end_idx]
            r_chunk = r[start_idx:end_idx]
            vartheta_chunk = vartheta[start_idx:end_idx]
            Dg_chunk = np.zeros(len(lon_chunk))

            print(f'Processing chunk {i + 1} of {n_chunks}...')
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=nmax, ellipsoid=ellipsoid)

            Dg_chunk = compute_gravity_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, Gm, r_chunk, Pnm_chunk, nmax, Dg_chunk)

            Dg[start_idx:end_idx] = Dg_chunk
            print('\n')

    Dg = Gm / r ** 2 * Dg * 10**5  # mGal
    
    return pd.Series(Dg)

@jit(nopython=True)
def compute_disturbance_chunk(Cnm, Snm, lon, a, GM, r, Pnm, n, dg):
    '''
    Compute a chunk of gravity disturbance for a specific degree n using Numba for optimization

    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D arrays)
    a        : Reference radius 
    GM       : Gravitational constant times the mass of the Earth
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    dg       : Gravity disturbance array to update

    Returns
    -------
    Updated dg array with computed values for degree n
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    dg += (n + 1) * (a / r) ** n * sum

    return dg

def compute_disturbance_for_chunk(Cnm, Snm, lon, a, GM, r, Pnm, nmax, dg):
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
    dg       : Gravity disturbance array to update

    Returns
    -------
    Updated dg array with computed values for all degrees
    '''
    with ProgressBar(total=nmax - 1, desc='Calculating Gravity Anomalies for chunk') as pbar:
        for n in range(2, nmax + 1):
            dg = compute_disturbance_chunk(Cnm, Snm, lon, a, GM, r, Pnm, n, dg)
            pbar.update(1)
    
    return dg

def gravity_disturbance(shc, grav_data=None, ellipsoid='wgs84', nmax=300, chunk_size=100, split_data=False):
    '''
    Wrapper function to handle data and call the Numba-optimized function
    
    Parameters
    ----------
    shc       : Spherical Harmonic Coefficients (output of icgem.read_icgem() and shtools.replace_zonal_harmonics())
    grav_data : Gravity data with columns lon, lat, and elevation: lat and lon units: degrees
    ellipsoid : Reference ellipsoid
    nmax      : Maximum spherical harmonic degree of expansion
    chunk_size: Size of the chunk of data to process at once
    split_data: Whether to split the data into chunks
    
    Returns
    -------
    dg        : Gravity disturbance (mGal)
    
    Notes
    -----
    1. Torge, Müller, & Pail (2023): Geodesy, Eq. 6.36b, p.297
    2. Please ensure that you have called shtools.replace_zonal_harmonics() on shc before passing it to gravity_disturbance()
    '''        
    if grav_data is None:
        raise ValueError('Provide data with columns lon, lat, and elevation in order')
    else:
        if isinstance(grav_data, np.ndarray):
            lon = grav_data[:, 0]
            lat = grav_data[:, 1]
            try:
                h = grav_data[:, 2]
            except IndexError:
                print('Looks like there is no elevation column. Setting elevation to 0')
                h = np.zeros(len(lat))
        elif isinstance(grav_data, pd.DataFrame):
            lon_column = [col for col in grav_data.columns if pd.Series(col).str.contains('lon', case=False).any()][0]
            lat_column = [col for col in grav_data.columns if pd.Series(col).str.contains('lat', case=False).any()][0]
            lon = grav_data[lon_column].values
            lat = grav_data[lat_column].values
            try:
                elev_column = [col for col in grav_data.columns if pd.Series(col).str.contains(r'elev|height', case=False).any()][0]
                h = grav_data[elev_column].values
            except IndexError:
                print('Looks like there is no elevation column. Setting elevation to 0')
                h = np.zeros(len(lat))
                
    r, vartheta, _ = co.geodetic2spherical(phi=lat, lambd=lon, height=h, ellipsoid=ellipsoid)
    
    Cnm = np.array(shc['Cnm'])
    Snm = np.array(shc['Snm'])
    a = shc['a']
    Gm = shc['GM']
    
    dg = np.zeros(len(lon))

    if not split_data:
        lon = np.radians(lon)
        Pnm = ALFsGravityAnomaly(vartheta=vartheta, nmax=nmax, ellipsoid=ellipsoid)
        dg = compute_disturbance_for_chunk(Cnm, Snm, lon, a, Gm, r, Pnm, nmax, dg)
    else:
        n_points = len(lon)
        n_chunks = (n_points // chunk_size) + 1
        print(f'Data will be processed in {n_chunks} chunks...\n')

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_points)

            lon_chunk = lon[start_idx:end_idx]
            r_chunk = r[start_idx:end_idx]
            vartheta_chunk = vartheta[start_idx:end_idx]
            dg_chunk = np.zeros(len(lon_chunk))

            print(f'Processing chunk {i + 1} of {n_chunks}...')
            Pnm_chunk = ALFsGravityAnomaly(vartheta=vartheta_chunk, nmax=nmax, ellipsoid=ellipsoid)

            dg_chunk = compute_disturbance_for_chunk(Cnm, Snm, np.radians(lon_chunk), a, Gm, r_chunk, Pnm_chunk, nmax, dg_chunk)

            dg[start_idx:end_idx] = dg_chunk
            print('\n')

    dg = Gm / r ** 2 * dg * 10**5  # mGal
    
    return pd.Series(dg)
