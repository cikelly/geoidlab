############################################################
# Utilities for converting between coordinate systems      #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    sin, cos, radians, 
    zeros, sqrt, arctan, tan,
    degrees
)
import copy

def geo_lat2geocentric(phi, ellipsoid='wgs84', semi_major=None, semi_minor=None):
    """
    Convert geodetic (geographic) latitude into geocentric latitude
    
    Parameters
    ----------
    phi       : geodetic latitude (degrees)
    semi_major: semi-major axis (a)
    semi_minor: semi-minor axis (b)
    ellipsoid : reference ellipsoid (wgs84 or grs80)
    
    Returns 
    -------
    phi_bar   : geocentric latitude (degrees)
    
    References
    ----------
    1. https://en.wikipedia.org/wiki/Geodetic_coordinates
    2. Physical Geodesy, Hofmann-Wellenhof and Moritz (2005)
    """
    if semi_major is None or semi_minor is None:
        ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
        a = ref_ellipsoid['semi_major']
        b = ref_ellipsoid['semi_minor']
    
    phi_bar = arctan((b/a)**2 * tan(radians(phi)))
    return degrees(phi_bar)