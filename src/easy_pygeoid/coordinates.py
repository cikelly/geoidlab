############################################################
# Utilities for converting between coordinate systems      #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
from . import constants
import numpy as np


def geodetic2geocentric(phi, ellipsoid='wgs84'):
    '''
    Convert geodetic (geographic) latitude into geocentric latitude
    
    Parameters
    ----------
    phi       : geodetic latitude (np.degrees)
    semi_major: semi-major axis (a)
    semi_minor: semi-minor axis (b)
    ellipsoid : reference ellipsoid (wgs84 or grs80)
    
    Returns 
    -------
    phi_bar   : geocentric latitude (np.degrees)
    
    References
    ----------
    1. https://en.wikipedia.org/wiki/Geodetic_coordinates
    2. Physical Geodesy, Hofmann-Wellenhof and Moritz (2005)
    '''
    # if semi_major is None or semi_minor is None:
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']
    b = ref_ellipsoid['semi_minor']
    
    phi_bar = np.arctan((b/a)**2 * np.tan(np.radians(phi)))
    return np.degrees(phi_bar)

def geodetic2cartesian(phi, lambd, ellipsoid, height=0):
    '''
    Estimate the radial distance from the center of an ellipsoid to a point
    
    Parameters
    ----------
    phi       : geodetic latitude (np.degrees)
    lambd     : geodetic longitude (np.degrees)
    height    : height above ellipsoid (m)
    
    Returns
    -------
    N         : radial distance from center of sphere to point
    X         : colatitude
    Y         : longitude (same as geodetic longitude)
    Z         :
    '''
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    a = ref_ellipsoid['semi_major']
    e2 = ref_ellipsoid['e2']
    
    phi = np.radians(phi)
    lambd = np.radians(lambd)
    
    N = a / np.sqrt( 1 - (e2*np.sin(phi)**2) )
    X = (N+height)*np.cos(phi)*np.cos(lambd)
    Y = (N+height)*np.cos(phi)*np.sin(lambd)
    # Z = (N+height)*np.sin(phi)
    Z = (N*(1-e2)+height)*np.sin(phi)
    
    return N, X, Y, Z
    
def geodetic2spherical(phi, lambd, ellipsoid, height=0):
    '''
    Estimate the radial distance from the center of an ellipsoid to a point
    
    Parameters
    ----------
    phi       : geodetic latitude (np.degrees)
    lambd     : geodetic longitude (np.degrees)
    height    : height above ellipsoid (m)
    
    Returns
    -------
    r         : radial distance from center of sphere to point
    vartheta  : colatitude
    lambda    : longitude (same as geodetic longitude)
    
    Notes
    -----
    1. We estimate the polar angle (psi) from the XY-plane to the point (X,Y,Z).
       This conversion is different from what you might see elsewhere, where
       the angle is taken from the positive Z-axis to the point (X,Y,Z) -- 
       example: https://en.wikipedia.org/wiki/Spherical_coordinate_system.
    2. Because of our conversion, we need to estimate colatitude (vartheta) as
       np.pi/2 - psi
    3. In the case where the polar angle is taken from the positive Z-axis to 
       the point, the polar angle is the same as the colatitude
    '''
    _, X, Y, Z = geodetic2cartesian(phi=phi, lambd=lambd, ellipsoid=ellipsoid, height=height)
    
    r = np.sqrt( X**2 + Y**2 + Z**2 )
    psi = np.arctan( Z/np.sqrt(X**2 + Y**2)) # polar angle from the XY-plane to the point (X,Y,Z)
    vartheta = np.pi/2 - psi # colatitude
    
    return r, vartheta, lambd
