############################################################
# Utilities for gravity modelling                          #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    sin, cos, radians, 
    zeros, sqrt, arctan, tan
)
import copy
from coordinates import geo_lat2geocentric

def normal_gravity(phi, ellipsoid='wgs84'):
    """
    Estimate normal gravity of a point on the reference ellipsoid

    Parameters
    ----------
    phi       : Geodetic latitude of the point (degrees)
    ellipsoid : Reference ellipsoid (wgs84 or grs80)

    Returns
    -------
    gamma_0   : normal gravity of the point on the ellipsoid

    Reference
    ---------
    Torge, Muller and Pail (2023), Geodesy (5th Edition)
    
    Notes
    -----
    1. Equation 4.41b, Page 147
    2. $a\gamma_b$ in Eq. 4.41b should be $a\gamma_a$
    3. This is correctly specified in the 4th Edition of the book.
    """
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
    gamma_0   : normal gravity of the point on the ellipsoid
    
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
    gamma_0_h : normal gravity of the point above the ellipsoid (mgal)
    '''
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']
    f = ref_ellipsoid['f']
    m = ref_ellipsoid['m']
    
    gamma_0 = normal_gravity(phi=phi, ellipsoid=ellipsoid)
    
    gamma_0_h = gamma_0 * (1 - 2/a*(1+f+m-2*f*sin(radians(phi))**2)*h + (3/a**2 * h**2))
    gamma_0_h = gamma_0_h * 1e5 # m/s2 to mgal
    return gamma_0_h


def ellipsoid_radius(phi, ellipsoid='wgs84'):
    """
    Define local ellipsoid radius (r)

    Parameters
    ----------
    phi       : geodetic latitude

    Returns
    -------
    r        : local ellipsoid radius
    """
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    semi_major = ref_ellipsoid['semi_major']

    e2 = ref_ellipsoid['e2']
    numerator = e2 * (1 - e2) * sin(radians(phi)) ** 2
    denominator = 1 - e2 * sin(radians(phi)) ** 2

    r = semi_major * sqrt( 1 - numerator / denominator )
    
    return r