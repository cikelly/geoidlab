############################################################
# Utilities for legendre polynomials                       #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    sin, cos, radians, 
    zeros, sqrt
)
from coordinates import geodetic2geocentric
 
def ALF(phi, nmax=60, ellipsoid='wgs84'):
    '''
    Compute associated Legendre functions

    Parameters
    ----------
    phi       : geodetic latitude (degrees)
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

    # Select ellipsoid parameters
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    semi_major = ref_ellipsoid['semi_major']
    semi_minor = ref_ellipsoid['semi_minor']

    # Convert geodetic latitude to geocentric latitude
    phi_bar = geodetic2geocentric(phi, semi_major=semi_major, semi_minor=semi_minor)
    # phi_bar = phi

    # Calculate trigonometric values
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
            a_nm = sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
            b_nm = 0.0
            if n - m - 1 >= 0:
                b_nm = sqrt((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0) / ((n - m) * (n + m) * (2.0 * n - 3.0)))
            Pnm[n, m] = a_nm * t * Pnm[n - 1, m] - b_nm * Pnm[n - 2, m]

        # Sectoral harmonics (n = m)
        Pnm[n, n] = u * sqrt((2.0 * n + 1.0) / (2.0 * n)) * Pnm[n - 1, n - 1]

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
    