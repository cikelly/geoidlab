############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from legendre import ALF
import coordinates as co
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
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    GMe = ref_ellipsoid['GM']
    a_e = ref_ellipsoid['semi_major']
    
    zonal_harmonics = ['C20', 'C40', 'C60', 'C80', 'C100']
    
    for n, Cn0 in zip([2, 4, 6, 8, 10], zonal_harmonics):
        shc['Cnm'][n,0] = shc['Cnm'][n,0] - ( shc['GM']/GMe) * (shc['a']/a_e )**2 * ref_ellipsoid[Cn0]
    
    # Fully normalized associated Legendre functions
    r, vartheta, _ = co.gedetic2spherical(phi=lat, lambd=lon, height=height, ellipsoid=ellipsoid)
    Pnm = ALF(phi=vartheta, nmax=nmax, ellipsoid=ellipsoid) 

    
    # Height anomaly
    

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