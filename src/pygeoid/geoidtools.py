############################################################
# Utilities for geoid modelling                            #
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

def mean_tide2zero_tide(N, phi, ellipsoid='wgs84'):
    """
    Convert geoid height from mean-tide system to zero-tide system

    Parameters
    ----------
    N         : numpy array of geoid heights
    varphi    : geodetic latitude (aka geographic latitude)

    Returns
    -------
    Nzero     : numpy array of geoid heights
    """

    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    semi_major = ref_ellipsoid['semi_major']
    semi_minor = ref_ellipsoid['semi_minor']

    varphi = geo_lat2geocentric(phi, semi_major, semi_minor)
    return N - ( -0.198 * (3/2 * sin(radians(varphi))**2 - 1/2) )


def mean_tide2free_tide(N, phi, ellipsoid='wgs84'):
    """
    Convert geoid height from mean-tide system into free-tide system

    Parameters
    ----------
    N         : numpy array of geoid heights
    varphi    : geodetic latitude (aka geographic latitude)

    Returns
    -------
    Nfree     : numpy array of geoid heights  

    Reference
    ---------
    Rapp 1989  
    """
    k = 0.3 # Love number

    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    semi_major = ref_ellipsoid['semi_major']
    semi_minor = ref_ellipsoid['semi_minor']

    varphi = geo_lat2geocentric(phi, semi_major, semi_minor)
    return N - ( (1+k) * (-0.198) * (3/2 * sin(radians(varphi))**2 - 1/2) )