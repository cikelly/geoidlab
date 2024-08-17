############################################################
# Utilities for gravity modelling                          #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    sin, cos, radians, 
    sqrt
)

from coordinates import geodetic2geocentric

def normal_gravity(phi, ellipsoid='wgs84'):
    r'''
    Estimate normal gravity of a point on the reference ellipsoid

    Parameters
    ----------
    phi       : Geodetic latitude of the point (degrees)
    ellipsoid : Reference ellipsoid (wgs84 or grs80)

    Returns
    -------
    gamma_0   : normal gravity of the point on the ellipsoid (m/s^2)

    Reference
    ---------
    Torge, Muller and Pail (2023), Geodesy (5th Edition)
    
    Notes
    -----
    1. Equation 4.41b, Page 147
    2. $a\gamma_b$ in Eq. 4.41b should be $a\gamma_a$
    3. This is correctly specified in the 4th Edition of the book.
    '''
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']
    b = ref_ellipsoid['semi_minor']

    gamma_a = ref_ellipsoid['gamma_a']
    gamma_b = ref_ellipsoid['gamma_b']
    e2      = ref_ellipsoid['e2']
           
    k       = b*gamma_b / (a*gamma_a) - 1

    numerator   = 1 + k * sin(radians(phi))**2
    denominator = sqrt(1 - e2 * sin(radians(phi))**2)
    
    gamma_0     = gamma_a * numerator / denominator
    return gamma_0

def normal_gravity_somigliana(phi, ellipsoid='wgs84'):
    '''
    Somigliana's formula for normal gravity on the ellipsoid

    Parameters
    ----------
    phi       : Geodetic latitude of the point (degrees)
    ellipsoid : Reference ellipsoid (wgs84 or grs80)

    Returns
    -------
    gamma_0   : normal gravity of the point on the ellipsoid (m/s^2)
    
    Reference
    ---------
    Torge, Muller and Pail (2023), Geodesy (5th Edition)
    
    Notes
    -----
    1. Equation 4.41a, Page 147
    '''
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']
    b = ref_ellipsoid['semi_minor']

    gamma_a = ref_ellipsoid['gamma_a']
    gamma_b = ref_ellipsoid['gamma_b']
    e2      = ref_ellipsoid['e2']
    
    numerator = a*gamma_a*cos(radians(phi))**2 + b*gamma_b*sin(radians(phi))**2
    denominator = sqrt(a**2*cos(radians(phi))**2 + b**2*sin(radians(phi))**2)
    
    gamma_0 = numerator / denominator
    return gamma_0


def normal_gravity_above_ellipsoid(phi, h, ellipsoid='wgs84'):
    '''
    Parameters
    ----------
    phi       : Geodetic latitude of the point (degrees)
    h         : Elevation of the point (m)
    ellipsoid : Reference ellipsoid (wgs84 or grs80)

    Returns
    -------
    gamma_h : normal gravity of a point above the ellipsoid (mgal)
    '''
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']
    f = ref_ellipsoid['f']
    m = ref_ellipsoid['m']
    
    gamma_0 = normal_gravity(phi=phi, ellipsoid=ellipsoid)
    
    gamma_h = gamma_0 * (1 - 2/a*(1+f+m-2*f*sin(radians(phi))**2)*h + (3/a**2 * h**2))
    gamma_h = gamma_h * 1e5 # m/s2 to mgal
    return gamma_h


def ellipsoid_radius(phi, ellipsoid='wgs84'):
    '''
    Define local ellipsoid radius (r)

    Parameters
    ----------
    phi       : geodetic latitude

    Returns
    -------
    r        : local ellipsoid radius
    '''
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']

    e2 = ref_ellipsoid['e2']
    numerator = e2 * (1 - e2) * sin(radians(phi)) ** 2
    denominator = 1 - e2 * sin(radians(phi)) ** 2

    r = a * sqrt( 1 - numerator / denominator )
    
    return r

def gravity_anomalies(lat, gravity, elevation, ellipsoid='wgs84'):
    '''
    Free-air and Bouguer gravity anomalies of a point on the Earth's 
    surface
    
    Parameters
    ----------
    lon              : longitude                (degrees)
    lat              : latitude                 (degrees)
    gravity          : gravity at the point     (m/s2)
    elevation        : height above the geoid   (m)
    ellipsoid        : Reference ellipsoid (wgs84 or grs80)
    
    Returns
    -------
    free_air_anomaly : free-air gravity anomaly (mgal)
    bouguer_anomaly  : Bouguer gravity anomaly (mgal)
    
    Notes
    -----
    1. Source for atm_corr: Encyclopedia of Geophysics Pg. 566
    2. Hofmann-Wellenhof & Moritz (2005): Physical Geodesy, 2nd edition, 
       Chapter 3, Equations 3-24 to 3-26 (free-air) and 3-28 to 3-29 (bouguer)
    '''
    # Free-air and Bouguer gravity
    free_air_gravity = gravity + 0.3086 * elevation
    bouguer_gravity  = free_air_gravity - 0.1119 * elevation
    
    # normal gravity
    gamma_0 = normal_gravity(phi=lat, ellipsoid=ellipsoid)
    gamma_0 = gamma_0 * 1e5 # m/s2 to mgal
    
    # Atmospheric correction
    atm_corr = 0.874 - 9.9e-5 * elevation + 3.56e-9*elevation**2

    # Gravity anomalies
    free_air_anomaly = free_air_gravity - gamma_0 + atm_corr
    bouguer_anomaly  = bouguer_gravity - gamma_0 + atm_corr
    
    return free_air_anomaly, bouguer_anomaly

def gravity_reduction(gravity, elevation):
    '''
    Free-air and Bouguer gravity reductions
    
    Parameters
    ----------
    gravity          : gravity at the point (m/s2)
    elevation        : height above the geoid (m)
    
    Returns
    -------
    free_air_gravity : free-air gravity (mgal)
    bouguer_gravity  : Bouguer gravity (mgal)
    '''
    # Free-air and Bouguer gravity
    free_air_gravity = gravity + 0.3086 * elevation
    bouguer_gravity = free_air_gravity - 0.1119 * elevation
    
    return free_air_gravity, bouguer_gravity
