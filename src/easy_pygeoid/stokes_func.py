############################################################
# Stokes' function and its modifications                   #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

from numpy import (
    sin, cos, radians, 
    sqrt, log, array,
    zeros_like, degrees
)

from legendre import legendre_poly

def stokes(comp_point, int_points) -> tuple:
    '''
    Calculate the Original Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    
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
    
def meissl(comp_point, int_points, psi_0) -> float:
    '''
    Calculate the Original Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    psi_0      : spherical distance of the spherical cap in radians
    
    Returns
    -------
    S         : Stokes' function
    '''
    S, _ = stokes(comp_point, int_points)
    S_0, _ = stokes([0, degrees(psi_0)], array([[0, 0]]))

    return S - S_0

def wong_and_gore(comp_point, int_points, nmax) -> float:
    '''
    Wong and Gore's modification of Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    nmax       : Maximum degree of expansion
    
    Returns
    -------
    S_wg      : Modified Stokes' function
    
    Notes
    -----
    1. Featherstone (2002): Software for computing five existing
       types of deterministically modified integration kernel for 
       gravimetric geoid determination
       https://www.sciencedirect.com/science/article/pii/S0098300402000742
    '''
    # Calculate original Stokes' function
    S, cos_psi = stokes(comp_point, int_points)
    
    # Wong and Gore's modification (Featherstone (2002): Eq. 21)
    S_wg = zeros_like(cos_psi)
    for i, t in enumerate(cos_psi):
        Pn = legendre_poly(t=t, nmax=nmax)
        sum_term = 0
        for n in range(2, nmax + 1):
            sum_term += (2 * n + 1) / (n - 1) * Pn[n]
        S_wg[i] = S[i] - sum_term
    
    return S_wg

def heck_and_gruninger(comp_point, int_points, psi_0, nmax) -> float:
    '''
    Heck and Gruninger's modification of Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    psi_0      : spherical distance of the spherical cap in radians
    nmax       : Maximum degree of expansion
    
    Returns
    -------
    S_hg      : Heck and Gruninger's modification of Stokes' function
    
    Notes
    -----
    1. Featherstone (2002): Software for computing five existing
       types of deterministically modified integration kernel for 
       gravimetric geoid determination
       https://www.sciencedirect.com/science/article/pii/S0098300402000742
    '''
    # Calculate original Stokes' function
    # S, cos_psi = stokes(comp_point, int_points)
    
    # Wong and Gore
    S_wg = wong_and_gore(comp_point, int_points, nmax)
    
    # Featherstone (2002): Eq. 26
    # Stokes' function for a spherical cap (psi_0)
    S_0, cos_psi_0 = stokes([0, degrees(psi_0)], array([[0, 0]]))
    # Wong and Gore for spherical cap (psi_0)
    t = cos_psi_0
    Pn = legendre_poly(t=t, nmax=nmax)
    S_wgL = 0
    for n in range(2, nmax + 1):
        S_wgL += (2 * n + 1) / (n - 1) * Pn[n]
    
    # Heck and Gruninger
    S_hg = S_wg - (S_0 - S_wgL)
    
    return S_hg


def stokes_psi(psi) -> tuple:
    '''
    Calculate the Original Stokes' function from spherical distance
    
    Parameters
    ----------
    psi       : array-like
                    Spherical distance in radians
    
    Returns
    -------
    S         : Stokes' function
    cos_psi   : Cosine of spherical distance
    '''
    cos_psi = cos(psi)
    
    # Calculate sin^2(psi/2) using the relationship sin^2(psi/2) = (1 - cos(psi)) / 2
    sin2_psi_2 = (1 - cos_psi) / 2
    
    # Calculate Stokes' function S
    S = 1 / sqrt(sin2_psi_2) - 6 * sqrt(sin2_psi_2) + 1 - 5 * cos_psi - \
        3 * cos_psi * log(sqrt(sin2_psi_2) + sin2_psi_2)
    
    return S, cos_psi


def wong_and_gore_psi(psi, nmax) -> float:
    '''
    Wong and Gore's modification of Stokes' function from spherical distance
    
    Parameters
    ----------
    psi       : array-like
                    Spherical distance in radians
    nmax      : Maximum degree of expansion
    
    Returns
    -------
    S_wg      : Modified Stokes' function
    '''
    # Calculate original Stokes' function
    S, cos_psi = stokes_psi(psi)
    
    # Wong and Gore's modification
    S_wg = zeros_like(cos_psi)
    for i, t in enumerate(cos_psi):
        Pn = legendre_poly(t=t, nmax=nmax)
        sum_term = 0
        for n in range(2, nmax + 1):
            sum_term += (2 * n + 1) / (n - 1) * Pn[n]
        S_wg[i] = S[i] - sum_term
    
    return S_wg

def heck_and_gruninger_psi(psi, psi0, nmax) -> float:
    '''
    Heck and Gruninger's modification of Stokes' function from spherical distance
    
    Parameters
    ----------
    psi       : array-like
                    Spherical distances in radians
    psi0      : float
                 Spherical distance of the spherical cap in radians
    nmax      : Maximum degree of expansion
    
    Returns
    -------
    S_hg      : Heck and Gruninger's modification of Stokes' function
    '''
    # Calculate Wong and Gore's modification for psi
    S_wg = wong_and_gore_psi(psi, nmax=nmax)
    S_wg0 = wong_and_gore_psi(array([psi0]), nmax=nmax)
    # Calculate original Stokes' function for psi0
    S_0, _ = stokes_psi(array([psi0]))
    
    # Heck and Gruninger's modification
    S_hg = S_wg - S_0[0] - S_wg0[0]
    
    return S_hg

def meissl_psi(psi, psi0) -> float:
    '''
    Meissl's modification of Stokes function
    
    Parameters
    ----------
    psi       : array-like, shape (n,)
                    Spherical distance in radians
    psi0      : float
                    Spherical distance of the spherical cap in radians
    
    Returns
    -------
    S_m       : Modified Stokes' function
    '''
    S, _ = stokes_psi(psi)
    S_w0 = stokes_psi(array(psi0))[0]
    
    return S - S_w0

# # For plotting purposes
# def stokes_func(sph_dist):
#     '''
#     Stokes' function for a given spherical distance
    
#     Parameters 
#     ----------
#     sph_dist  : spherical distance
    
#     Returns
#     -------
#     S         : Stokes' function
    
#     Notes
#     ---------
#     1. Physical Geodesy (2nd Edition) Page 104, Equation 2–305
#     2. For numerical efficiency and accuracy, we will use a slightly modified form of Equation 2-305
#     '''    
#     S = 1/sin(sph_dist/2) - 6*sin(sph_dist/2) + 1 - 5*cos(sph_dist) - \
#         3*cos(sph_dist)*log(sin(sph_dist/2) + sin(sph_dist/2)**2)
    
#     return S
