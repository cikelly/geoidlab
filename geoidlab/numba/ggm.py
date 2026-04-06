############################################################
# Utilities for numba optimized GGM synthesis              #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################
import numpy as np
from numba import njit, prange

@njit
def compute_gravity_chunk(Cnm, Snm, lon, a, r, Pnm, n) -> np.ndarray:
    '''
    Compute gravity anomaly for a specific degree using Numba optimization.
    
    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    
    Returns
    -------
    Dg_chunk : Gravity anomaly contribution for degree n (1D array)
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    return (n - 1) * (a / r) ** n * sum


@njit(parallel=True)
def compute_gravity_chunk_parallel(Cnm, Snm, lon, a, r, Pnm, nmax) -> np.ndarray:
    '''
    Compute gravity anomaly for an entire chunk using point-wise threaded Numba.

    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    nmax     : Maximum spherical harmonic degree

    Returns
    -------
    Dg_chunk : Gravity anomaly contribution for the chunk (1D array)
    '''
    num_points = len(lon)
    Dg_chunk = np.zeros(num_points)

    for i in prange(num_points):
        total = 0.0
        lon_i = lon[i]
        r_i = r[i]
        for n in range(2, nmax + 1):
            degree_sum = 0.0
            for m in range(n + 1):
                degree_sum += (
                    Cnm[n, m] * np.cos(m * lon_i) +
                    Snm[n, m] * np.sin(m * lon_i)
                ) * Pnm[i, n, m]
            total += (n - 1) * (a / r_i) ** n * degree_sum
        Dg_chunk[i] = total

    return Dg_chunk

@njit
def compute_disturbance_chunk(Cnm, Snm, lon, a, r, Pnm, n) -> np.ndarray:
    '''
    Compute gravity disturbance for a specific degree using Numba optimization.
    
    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    
    Returns
    -------
    dg_chunk : Gravity anomaly contribution for degree n (1D array)
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    return (n + 1) * (a / r) ** n * sum


@njit
def compute_disturbing_potential_chunk(Cnm, Snm, lon, a, r, Pnm, n) -> np.ndarray:
    '''
    Compute disturbing potential for a specific degree using Numba optimization.
    
    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    
    Returns
    -------
    T_chunk : Gravity anomaly contribution for degree n (1D array)
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    return (a / r) ** n * sum

@njit
def compute_disturbing_potential_derivative_chunk(Cnm, Snm, lon, a, r, dPnm, n) -> np.ndarray:
    '''
    Compute the derivative of disturbing potential for a specific degree using Numba optimization.
    
    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    dPnm     : Derivative of Associated Legendre functions (3D array)
    n        : Specific degree
    
    Returns
    -------
    dT_dtheta_chunk : Contribution to dT/dtheta for degree n (1D array)
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * dPnm[:, n, m]
    return (a / r) ** n * sum

@njit
def compute_second_radial_chunk(Cnm, Snm, lon, a, r, Pnm, n) -> np.ndarray:
    '''
    Compute disturbing potential for a specific degree using Numba optimization.
    
    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    
    Returns
    -------
    T_chunk : Gravity anomaly contribution for degree n (1D array)
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    return (n + 1) * (n + 2) * (a / r) ** n * sum

@njit
def compute_separation_chunk(Cnm, Snm, lon, a, r, Pnm, n) -> np.ndarray:
    '''
    Compute geoid-quasi geoid separation for a specific degree using Numba optimization.
    
    Parameters
    ----------
    Cnm, Snm : Spherical Harmonic Coefficients (2D arrays)
    lon      : Longitude (1D array, radians)
    a        : Reference radius
    r        : Radial distance (1D array)
    Pnm      : Associated Legendre functions (3D array)
    n        : Specific degree
    
    Returns
    -------
    H_chunk : Separation contribution for degree n (1D array)
    '''
    sum = np.zeros(len(lon))
    for m in range(n + 1):
        sum += (Cnm[n, m] * np.cos(m * lon) + Snm[n, m] * np.sin(m * lon)) * Pnm[:, n, m]
    return sum
