############################################################
# Utilities for legendre polynomials                       #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
# import constants
from geoidlab.coordinates import geodetic2spherical
from geoidlab.numba.legendre import compute_legendre_chunk
from numba_progress import ProgressBar

from typing import Tuple
import numpy as np
 
def ALF(
    phi=None, 
    lambd=None, 
    vartheta=None, 
    height=None, 
    nmax=60, 
    ellipsoid='wgs84'
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    '''
    Compute fully normalized associated Legendre functions and their first derivatives

    Parameters
    ----------
    phi                : geodetic latitude (degrees)
    lambd              : geodetic longitude (degrees)
    vartheta           : colatitude (radians)
    height             : height (m)
    nmax               : maximum degree of expansion
    ellipsoid          : reference ellipsoid ('wgs84' or 'grs80')
    compute_derivative : if True, compute the derivative of the associated Legendre functions
    
    
    Returns
    -------
    Pnm                : Fully normalized Associated Legendre functions
    dPnm               : First derivative of the ALFs with respect to colatitude
    
    References
    ----------
    (1) Holmes and Featherstone (2002): A unified approach to the Clenshaw 
    summation and the recursive computation of very high degree and order 
    normalised associated Legendre functions (Eqs. 11 and 12)
    (2) Colombo (1981): Numerical Methods for Harmonic Analysis on the Sphere
    '''
    if phi is None and vartheta is None:
        raise ValueError('Either phi or vartheta must be provided')
    
    if lambd is None and vartheta is None:
        raise ValueError('Please provide lambd')
    
    if vartheta is not None: 
        phi_bar = vartheta
    elif phi is not None:
        if height is None:
            height = 0
        _, phi_bar, _ = geodetic2spherical(phi, lambd, ellipsoid, height=height)
        # phi_bar = degrees(phi_bar)
    
    # sine (u) and cosine (t) terms
    t = np.cos(phi_bar)
    u = np.sin(phi_bar)
    u = np.where(u == 0, np.finfo(float).eps, u) # avoid division by zero
    
    # Initialize the Pnm and dPnm arrays
    Pnm = np.zeros((nmax + 1, nmax + 1))
    dPnm = np.zeros((nmax + 1, nmax + 1))
    
    Pnm[0, 0] = 1.0
    dPnm[0, 0] = 0.0

    # Initialize first few values
    if nmax >= 1:
        Pnm[1, 0] = np.sqrt(3.0) * t
        Pnm[1, 1] = np.sqrt(3.0) * u
        dPnm[1, 0] = (1.0 / u) * (t * Pnm[1, 0] - np.sqrt(3.0) * Pnm[0, 0])
        dPnm[1, 1] = (t / u) * Pnm[1, 1]

    # Recursive computation of Pnm
    for n in range(2, nmax + 1):
        for m in range(0, n):
            a_nm = np.sqrt((2. * n - 1.) * (2. * n + 1.0) / ((n - m) * (n + m)))
            b_nm = 0.
            if n - m - 1 >= 0:
                b_nm = np.sqrt((2. * n + 1.) * (n + m - 1.) * (n - m - 1.) / ((n - m) * (n + m) * (2. * n - 3.)))
            Pnm[n, m] = a_nm * t * Pnm[n - 1, m] - b_nm * Pnm[n - 2, m]
            fnm = np.sqrt((n**2.0 - m**2.0) * (2.0 * n + 1.0) / (2.0 * n - 1.0))
            dPnm[n, m] = (1.0 / u) * (n * t * Pnm[n, m] - fnm * Pnm[n - 1, m])

        # Sectoral harmonics (n = m)
        Pnm[n, n] = u * np.sqrt((2. * n + 1.) / (2. * n)) * Pnm[n - 1, n - 1]
        dPnm[n, n] = n * (t / u) * Pnm[n, n]

    return Pnm, dPnm
    
def legendre_poly(theta=None, t=None, nmax=60) -> np.ndarray:
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
        t = np.cos(np.radians(theta))
        # if not -90 <= theta <= 90:
        #     raise ValueError('theta must be in the range [-90, 90]')
        # t = np.cos(np.radians(theta))
    # elif t is not None:
    #     if not -1 <= t <= 1:
    #         raise ValueError('t must be in the range [-1, 1]')
    
    # t     = np.cos(radians(theta))
    Pn    = np.zeros((nmax+1,))
    Pn[0] = 1
    Pn[1] = t

    for n in range(2, nmax+1):
        Pn[n] = ( (2*n-1)*t*Pn[n-1] - (n-1)*Pn[n-2] ) / n
    
    return Pn

def legendre_poly_fast(theta=None, t=None, nmax=60) -> np.ndarray:
    '''
    Compute Legendre polynomials of the First Kind (m=0) for scalar or array inputs.

    Parameters
    ----------
    theta     : geodetic latitude (degrees), scalar or array
    t         : cosine of theta, scalar or array
    nmax      : maximum degree of expansion

    Returns 
    -------
    Pn        : Legendre polynomials, 1D array (nmax+1,) if t is scalar, 
                2D array (len(t), nmax+1) if t is array
    '''
    if theta is None and t is None:
        raise ValueError('Either theta or t must be provided')
    
    if theta is not None:
        t = np.cos(np.radians(theta))
    
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = np.asarray(t).ravel()
    
    N = len(t)
    Pn = np.zeros((N, nmax + 1))
    Pn[:, 0] = 1
    if nmax >= 1:
        Pn[:, 1] = t
    for n in range(2, nmax + 1):
        Pn[:, n] = ((2 * n - 1) * t * Pn[:, n - 1] - (n - 1) * Pn[:, n - 2]) / n
    
    return Pn[0] if N == 1 else Pn



def ALFsGravityAnomaly(
    phi=None, 
    lambd=None, 
    height=None, 
    vartheta=None, 
    nmax=60, 
    ellipsoid='wgs84',
    show_progress=True
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Wrapper function to handle data and call the Numba-optimized function

    Parameters
    ----------
    phi           : geodetic latitude (degrees)
    lambd         : geodetic longitude (degrees)
    vartheta      : colatitude (radians)
    nmax          : maximum degree of expansion
    ellipsoid     : reference ellipsoid ('wgs84' or 'grs80')
    show_progress : show progress bar
    
    Returns
    -------
    Pnm           : Fully normalized Associated Legendre functions
    dPnm          : First derivative of Associated Legendre functions
    '''
    if phi is None and vartheta is None:
        raise ValueError('Either phi or vartheta must be provided')
    
    if lambd is None and vartheta is None:
        raise ValueError('Please provide lambd')
    
    if vartheta is not None: 
        phi_bar = vartheta
    elif phi is not None:
        if height is None:
            height = 0
        _, phi_bar, _ = geodetic2spherical(phi=phi, lambd=lambd, ellipsoid=ellipsoid, height=height)
    
    # Initialize Pnm array
    Pnm = np.zeros((len(phi_bar), nmax + 1, nmax + 1))
    dPnm = np.zeros((len(phi_bar), nmax + 1, nmax + 1))
    Pnm[:, 0, 0] = 1.0

    if nmax >= 1:
        t = np.cos(phi_bar)
        u = np.sin(phi_bar)
        u = np.where(u == 0, np.finfo(float).eps, u) # avoid division by zero
        Pnm[:, 1, 0] = np.sqrt(3.0) * t
        Pnm[:, 1, 1] = np.sqrt(3.0) * u
        dPnm[:, 1, 0] = (1.0 / u) * (t * Pnm[:, 1, 0] - np.sqrt(3.0) * Pnm[:, 0, 0])
        dPnm[:, 1, 1] = (t / u) * Pnm[:, 1, 1]

    # Initialize progress bar
    if show_progress:
        with ProgressBar(total=nmax - 1, desc='Computing Legendre Functions') as pbar:
            for n in range(2, nmax + 1):
                Pnm, dPnm = compute_legendre_chunk_with_deriv(phi_bar, n, Pnm, dPnm)
                pbar.update(1)
    else:
        # print('Computing Legendre Functions...')
        for n in range(2, nmax + 1):
            Pnm, dPnm = compute_legendre_chunk_with_deriv(phi_bar, n, Pnm, dPnm)
    
    return Pnm, dPnm

def compute_legendre_chunk_with_deriv(phi_bar, n, Pnm, dPnm) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute associated Legendre functions and derivatives for a specific degree n

    Parameters
    ----------
    phi_bar   : colatitude (radians), array-like
    n         : degree of expansion
    Pnm       : Associated Legendre functions array, shape (len(phi_bar), nmax+1, nmax+1)
    dPnm      : Derivatives array, shape (len(phi_bar), nmax+1, nmax+1)
    
    Returns
    -------
    Pnm       : Updated Associated Legendre functions
    dPnm      : Updated derivatives
    '''
    t = np.cos(phi_bar)
    u = np.sin(phi_bar)
    u = np.where(u == 0, np.finfo(float).eps, u)

    for m in range(0, n):
        a_nm = np.sqrt((2. * n - 1.) * (2. * n + 1.0) / ((n - m) * (n + m)))
        b_nm = 0.
        if n - m - 1 >= 0:
            b_nm = np.sqrt((2. * n + 1.) * (n + m - 1.) * (n - m - 1.) / ((n - m) * (n + m) * (2. * n - 3.)))
        Pnm[:, n, m] = a_nm * t * Pnm[:, n - 1, m] - b_nm * Pnm[:, n - 2, m]
        f_nm = np.sqrt((n**2.0 - m**2.0) * (2.0 * n + 1.0) / (2.0 * n - 1.0))
        dPnm[:, n, m] = (1.0 / u) * (n * t * Pnm[:, n, m] - f_nm * Pnm[:, n - 1, m])

    # Sectoral harmonics (n = m)
    Pnm[:, n, n] = u * np.sqrt((2. * n + 1.) / (2. * n)) * Pnm[:, n - 1, n - 1]
    dPnm[:, n, n] = n * (t / u) * Pnm[:, n, n]

    return Pnm, dPnm

# TO DO: Add derivative of Legendre polynomial

# def legendre_deriv(n, vartheta, ellipsoid='wgs84'):
#     '''
#     Compute the derivative of the associated Legendre functions

#     Parameters
#     ----------
#     n         : specific degree
#     vartheta  : colatitude (radians)
#     ellipsoid : reference ellipsoid ('wgs84' or 'grs80')
    
#     Returns
#     -------
#     Pnm       : Fully normalized Associated Legendre functions
#     '''
#     Pnm = ALFsGravityAnomaly(vartheta=vartheta, nmax=n, ellipsoid=ellipsoid)
#     Pnm = Pnm * n

#     return Pnm
