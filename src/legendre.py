############################################################
# Utilities for legendre polynomials                       #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    sin, cos, radians, 
    zeros, sqrt, degrees
)
from coordinates import geodetic2spherical
 
def ALF(phi=None, lambd=None, vartheta=None, nmax=60, ellipsoid='wgs84'):
    '''
    Compute associated Legendre functions

    Parameters
    ----------
    phi       : geodetic latitude (degrees)
    vartheta  : colatitude (degrees)
    lambd     : geodetic longitude (degrees)
    nmax      : maximum degree of expansion
    ellipsoid : reference ellipsoid ('wgs84' or 'grs80')
    
    Returns
    -------
    Pnm       : Fully normalized Associated Legendre functions
    
    References
    ----------
    (1) Holmes and Featherstone (2002): A unified approach to the Clenshaw 
    summation and the recursive computation of very high degree and order 
    normalised associated Legendre functions (Eqs. 11 and 12)
    '''
    if phi is None and vartheta is None:
        raise ValueError('Either phi or vartheta must be provided')
    
    if lambd is None and vartheta is None:
        raise ValueError('Please provide lambd')
    
    if vartheta is not None: 
        phi_bar = vartheta
    elif phi is not None:
        _, phi_bar, _ = geodetic2spherical(phi, lambd, ellipsoid, height=0)
        phi_bar = degrees(phi_bar)
    
    # sine (u) and cosine (t) terms
    t = cos(radians(phi_bar))
    u = sin(radians(phi_bar))

    # Initialize the Pnm array
    Pnm = zeros((nmax + 1, nmax + 1))
    Pnm[0, 0] = 1.0

    # Initialize first few values
    if nmax >= 1:
        Pnm[1, 0] = sqrt(3.0) * t
        Pnm[1, 1] = sqrt(3.0) * u

    # Recursive computation of Pnm
    for n in range(2, nmax + 1):
        for m in range(0, n):
            a_nm = sqrt((2. * n - 1.) * (2. * n + 1.0) / ((n - m) * (n + m)))
            b_nm = 0.
            if n - m - 1 >= 0:
                b_nm = sqrt((2. * n + 1.) * (n + m - 1.) * (n - m - 1.) / ((n - m) * (n + m) * (2. * n - 3.)))
            Pnm[n, m] = a_nm * t * Pnm[n - 1, m] - b_nm * Pnm[n - 2, m]

        # Sectoral harmonics (n = m)
        Pnm[n, n] = u * sqrt((2. * n + 1.) / (2. * n)) * Pnm[n - 1, n - 1]

    return Pnm
    
def legendre_poly(theta=None, t=None, nmax=60):
    '''
    Compute Legendre polynomials of the First Kind i.e., m=0

    Parameters
    ----------
    theta     : geodetic latitude (degrees)
    nmax      : maximum degree of expansion
    t         : cosine of theta 
    
    Returns 
    -------
    Pn        : Legendre polynomials
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Legendre_polynomials
    '''
    if theta is None and t is None:
        raise ValueError('Either theta or t must be provided')
    
    if theta is not None:
        if not -90 <= theta <= 90:
            raise ValueError('theta must be in the range [-90, 90]')
        t = cos(radians(theta))
    elif t is not None:
        if not -1 <= t <= 1:
            raise ValueError('t must be in the range [-1, 1]')
    
    # t     = cos(radians(theta))
    Pn    = zeros((nmax+1,))
    Pn[0] = 1
    Pn[1] = t

    for n in range(2, nmax+1):
        Pn[n] = ( (2*n-1)*t*Pn[n-1] - (n-1)*Pn[n-2] ) / n
    
    return Pn


# def ALF(phi=None, vartheta=None, nmax=60, ellipsoid='wgs84'):
#     '''
#     Compute associated Legendre functions

#     Parameters
#     ----------
#     phi       : geodetic latitude (degrees)
#     vartheta  : colatitude        (degrees)
#     nmax      : maximum degree of expansion
#     ellipsoid : reference ellipsoid ('wgs84' or 'grs80')
    
#     Returns
#     -------
#     Pnm       : Fully normalized Associated Legendre functions
    
#     References
#     ----------
#     (1) Holmes and Featherstone (2002): A unified approach to the Clenshaw 
#     summation and the recursive computation of very high degree and order 
#     normalised associated Legendre functions (Eqs. 11 and 12)
#     '''

#     if phi is None and vartheta is None:
#         raise ValueError('Either phi or vartheta must be provided')

#     if phi is not None and vartheta is not None: 
#         phi_bar = vartheta
#     elif phi is not None:
#         phi_bar = geodetic2geocentric(phi, ellipsoid=ellipsoid)
#     elif vartheta is not None:
#         phi_bar = vartheta

#     # Calculate trigonometric values
#     t = cos(radians(phi_bar))
#     u = sin(radians(phi_bar))

#     # Initialize the Pnm array
#     Pnm = zeros((nmax + 1, nmax + 1))
#     Pnm[0, 0] = 1.0

#     # Initialize first few values
#     if nmax >= 1:
#         Pnm[1, 0] = sqrt(3.0) * t
#         Pnm[1, 1] = sqrt(3.0) * u

#     # Recursive computation of Pnm
#     for n in range(2, nmax + 1):
#         for m in range(0, n):
#             a_nm = sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
#             b_nm = 0.0
#             if n - m - 1 >= 0:
#                 b_nm = sqrt((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0) / ((n - m) * (n + m) * (2.0 * n - 3.0)))
#             Pnm[n, m] = a_nm * t * Pnm[n - 1, m] - b_nm * Pnm[n - 2, m]

#         # Sectoral harmonics (n = m)
#         Pnm[n, n] = u * sqrt((2.0 * n + 1.0) / (2.0 * n)) * Pnm[n - 1, n - 1]

#     return Pnm
    