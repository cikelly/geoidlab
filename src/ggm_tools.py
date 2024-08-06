############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from legendre import ALF, ALFsGravityAnomaly
from shtools import replace_zonal_harmonics

import coordinates as co
import numpy as np
import pandas as pd

import gravity
import tide_system
import icgem
import constants



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
    
    
    gamma_0 = gravity.normal_gravity(phi=lat, ellipsoid=ellipsoid)
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
    
    
    gamma_0 = gravity.normal_gravity(phi=lat, ellipsoid=ellipsoid)
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
    
    gamma_0 = gravity.normal_gravity(phi=lat, ellipsoid=ellipsoid)
    
    N = geoid + ( (GM - GMe) / R - (W0 - U0) ) / gamma_0 
    
    return N
    
def gravity_anomaly(shc, grav_data=None, ellipsoid='wgs84', nmax=300):
    '''
    Calculate gravity anomalies from global model
    
    Parameters
    ----------
    shc       : Spherical Harmonic Coefficients (output of icgem.read_icgem() and shtools.replace_zonal_harmonics())
    grav_data : Gravity data with columns lon, lat, and elevation: lat and lon units: degrees
    ellipsoid : Reference ellipsoid
    nmax      : Maximum spherical harmonic degree of expansion
    
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
            lon = grav_data[:,0]
            lat = grav_data[:,1]
            try:
                h = grav_data[:,2]
            except IndexError:
                print('Looks like there is no elevation column. Setting elevation to 0')
                h = np.zeros(len(lat))
        elif isinstance(grav_data, pd.DataFrame):
            lon_column = [col for col in grav_data.columns if pd.Series(col).str.contains('lon', case=False).any()][0]
            lat_column = [col for col in grav_data.columns if pd.Series(col).str.contains('lat', case=False).any()][0]
            lon = grav_data[lon_column]
            lat = grav_data[lat_column]
            try:
                elev_column = [col for col in grav_data.columns if pd.Series(col).str.contains(r'elev|height', case=False).any()][0]
                h = grav_data[elev_column]
            except IndexError:
                print('Looks like there is no elevation column. Setting elevation to 0')
                h = np.zeros(len(lat))
                
    r, vartheta, _ = co.geodetic2spherical(phi=lat, lambd=lon, height=h, ellipsoid=ellipsoid)
    
    # if len(grav_data) > 1:
    #     Pnm = np.zeros((len(vartheta), nmax+1, nmax+1))
        
    lon = np.radians(lon)
    lat = np.radians(lat)
    
    Pnm = ALFsGravityAnomaly(vartheta=vartheta, nmax=nmax, ellipsoid=ellipsoid)
    
    # Gravity anomalies
    Dg = np.zeros(len(grav_data))
        
    for n in range(1, nmax+1):
        sum = np.zeros(len(grav_data))
        for m in range(n+1):
            # sum += (shc['Cnm'][n, m] * np.cos(m*lon)) + (shc['Snm'][n, m] * np.sin(m*lon)) * Pnm[n, m]
            sum += (shc['Cnm'][n, m] * np.cos(m*lon)) + (shc['Snm'][n, m] * np.sin(m*lon)) * Pnm[:, n, m]
        Dg += (n-1) * (shc['a'] / r) ** n * sum
    
    Dg = shc['GM'] / r ** 2 * Dg * 10**5 # mGal
    
    return Dg