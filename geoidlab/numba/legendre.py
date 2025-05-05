############################################################
# Utilities for numba optimized legendre polynomials       #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import numpy as np
from numba import njit, jit

@jit(nopython=True)
def compute_legendre_chunk(vartheta, n, Pnm) -> np.ndarray:
    '''
    Compute a chunk of associated Legendre functions for a specific degree n using Numba for optimization

    Parameters
    ----------
    vartheta  : colatitude (radians)
    n         : specific degree
    Pnm       : array to store the computed Legendre functions

    Returns
    -------
    Updated Pnm array with computed values for degree n
    '''
    t = np.cos(vartheta)
    u = np.sin(vartheta)

    for m in range(0, n):
        a_nm = np.sqrt((2. * n - 1.) * (2. * n + 1.0) / ((n - m) * (n + m)))
        b_nm = 0.
        if n - m - 1 >= 0:
            b_nm = np.sqrt((2. * n + 1.) * (n + m - 1.) * (n - m - 1.) / ((n - m) * (n + m) * (2. * n - 3.)))
        Pnm[:, n, m] = a_nm * t * Pnm[:, n - 1, m] - b_nm * Pnm[:, n - 2, m]
    # Sectoral harmonics (n = m)
    Pnm[:, n, n] = u * np.sqrt((2. * n + 1.) / (2. * n)) * Pnm[:, n - 1, n - 1]

    return Pnm

@njit(parallel=True)
def leg_poly_numba(t, nmax) -> np.ndarray:
    '''
    Compute Legendre polynomials using Numba for optimization

    Parameters
    ----------
    t         : cos(theta)
    nmax      : maximum degree of expansion

    Returns
    -------
    Pn        : Legendre polynomials
    '''
    Pn = np.zeros(nmax + 1)
    Pn[0] = 1.0
    if nmax >= 1:
        Pn[1] = t
    for n in range(2, nmax + 1):
        Pn[n] = ((2 * n - 1) * t * Pn[n-1] - (n - 1) * Pn[n-2]) / n
    return Pn

def legendre_poly_numba(theta=None, t=None, nmax=60) -> np.ndarray:
    '''
    Wrapper function to handle data and call the Numba-optimized leg_poly_numba

    Parameters
    ----------
    theta    : colatitude (degrees)
    t        : cos(theta)
    nmax     : maximum degree of expansion

    Returns
    -------
    Pn       : Legendre polynomials
    '''
    if theta is None and t is None:
        raise ValueError('Either theta or t must be provided')
    
    # If theta is provided, overwrite t
    if theta is not None:
        t = np.cos(np.radians(theta))
        
    return leg_poly_numba(t, nmax)