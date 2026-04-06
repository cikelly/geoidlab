############################################################
# Utilities for numba optimized legendre polynomials       #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import numpy as np
from numba import njit, jit, prange

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
def compute_legendre_chunk_parallel(vartheta, n, Pnm) -> np.ndarray:
    '''
    Compute a chunk of associated Legendre functions for a specific degree n
    using point-wise threaded Numba.

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

    num_points = len(vartheta)
    for i in prange(num_points):
        for m in range(0, n):
            a_nm = np.sqrt((2. * n - 1.) * (2. * n + 1.0) / ((n - m) * (n + m)))
            b_nm = 0.
            if n - m - 1 >= 0:
                b_nm = np.sqrt((2. * n + 1.) * (n + m - 1.) * (n - m - 1.) / ((n - m) * (n + m) * (2. * n - 3.)))
            Pnm[i, n, m] = a_nm * t[i] * Pnm[i, n - 1, m] - b_nm * Pnm[i, n - 2, m]
        Pnm[i, n, n] = u[i] * np.sqrt((2. * n + 1.) / (2. * n)) * Pnm[i, n - 1, n - 1]

    return Pnm


@njit(parallel=True)
def compute_legendre_chunk_with_deriv_parallel(phi_bar, n, Pnm, dPnm):
    '''
    Compute associated Legendre functions and derivatives for a specific degree n
    using point-wise threaded Numba.

    Parameters
    ----------
    phi_bar   : colatitude (radians), array-like
    n         : degree of expansion
    Pnm       : Associated Legendre functions array
    dPnm      : Derivatives array

    Returns
    -------
    Pnm       : Updated Associated Legendre functions
    dPnm      : Updated derivatives
    '''
    t = np.cos(phi_bar)
    u = np.sin(phi_bar)

    num_points = len(phi_bar)
    for i in prange(num_points):
        u_i = u[i]
        if u_i == 0.0:
            u_i = np.finfo(np.float64).eps
        t_i = t[i]

        for m in range(0, n):
            a_nm = np.sqrt((2. * n - 1.) * (2. * n + 1.0) / ((n - m) * (n + m)))
            b_nm = 0.
            if n - m - 1 >= 0:
                b_nm = np.sqrt((2. * n + 1.) * (n + m - 1.) * (n - m - 1.) / ((n - m) * (n + m) * (2. * n - 3.)))
            Pnm[i, n, m] = a_nm * t_i * Pnm[i, n - 1, m] - b_nm * Pnm[i, n - 2, m]
            f_nm = np.sqrt((n**2.0 - m**2.0) * (2.0 * n + 1.0) / (2.0 * n - 1.0))
            dPnm[i, n, m] = (1.0 / u_i) * (n * t_i * Pnm[i, n, m] - f_nm * Pnm[i, n - 1, m])

        Pnm[i, n, n] = u_i * np.sqrt((2. * n + 1.) / (2. * n)) * Pnm[i, n - 1, n - 1]
        dPnm[i, n, n] = n * (t_i / u_i) * Pnm[i, n, n]

    return Pnm, dPnm

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
