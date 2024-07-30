############################################################
# Stokes' function and its modifications                   #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from numpy import (
    sin, cos, radians, 
    sqrt, log, array
)

def stokes(comp_point, int_points):
    '''
    Calculate the Original Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points
    
    Returns
    -------
    S         : Stokes' function
    cos_psi   : Cosine of spherical distance
    
    Notes
    -----
    1. https://en.wikipedia.org/wiki/Haversine_formula
    2. cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
    3. Spherical cosines
    4. Estimate sin2_psi_2 = sin^2(psi/2) using Haversine formula (Note 1)
    5. Physical Geodesy (2nd Edition) Page 104, Equation 2–305
    '''
    lonp, latp = array(comp_point)
    int_points = array(int_points) if isinstance(int_points, list) else int_points
    lon, lat = int_points[:, 0], int_points[:, 1]
    
    lon, lat, lonp, latp = radians(lon), radians(lat), radians(lonp), radians(latp)
    
    # Calculate cos_dlam using spherical trigonometry (Note 2)
    cos_dlam = cos(lon) * cos(lonp) + sin(lon) * sin(lonp)  
    
    # Calculate cos_psi using the spherical law of cosines (Note 3)
    cos_psi = sin(latp) * sin(lat) + cos(latp) * cos(lat) * cos_dlam  
    
    # Calculate sin^2(psi/2) using the Haversine formula (Note 4)
    sin2_psi_2 = sin( (latp - lat)/2 )**2 + cos(latp) * cos(lat) * sin( (lonp - lon)/2 )**2 
    
    # Calculate Stokes' function S (Note 5)
    S = 1/sqrt(sin2_psi_2) - 6*sqrt(sin2_psi_2) + 1 - 5*cos_psi - \
        3*cos_psi*log(sqrt(sin2_psi_2) + sin2_psi_2)
        
    return S, cos_psi
    
    
def stokes_func(sph_dist):
    '''
    Stokes' function for a given spherical distance
    
    Parameters 
    ----------
    sph_dist  : spherical distance
    
    Returns
    -------
    S         : Stokes' function
    
    Notes
    ---------
    1. Physical Geodesy (2nd Edition) Page 104, Equation 2–305
    2. For numerical efficiency and accuracy, we will use a slightly modified form of Equation 2-305
    '''    
    S = 1/sin(sph_dist/2) - 6*sin(sph_dist/2) + 1 - 5*cos(sph_dist) - \
        3*cos(sph_dist)*log(sin(sph_dist/2) + sin(sph_dist/2)**2)
    
    return S
    

def wong_and_gore():
    '''
    '''
    pass

def heck_and_gruninger():
    '''
    '''
    
    
