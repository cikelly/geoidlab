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


@njit
def compute_alfs_holmes_pointwise(phi_bar, nmax, compute_derivative):
    '''
    Compute Holmes-Featherstone ALFs point-by-point with compiled loops.
    '''
    n_points = len(phi_bar)
    tri_size = (nmax + 1) * (nmax + 2) // 2
    scalef = 1.0e-280

    f1 = np.zeros(tri_size, dtype=np.float64)
    f2 = np.zeros(tri_size, dtype=np.float64)
    plm = np.zeros((n_points, nmax + 1, nmax + 1), dtype=np.float64)
    dplm = np.zeros((n_points, nmax + 1, nmax + 1), dtype=np.float64) if compute_derivative else np.zeros((1, 1, 1), dtype=np.float64)

    k = 2
    for l in range(2, nmax + 1):
        k += 1
        f1[k] = np.sqrt(2.0 * l - 1.0) * np.sqrt(2.0 * l + 1.0) / l
        f2[k] = (l - 1.0) * np.sqrt(2.0 * l + 1.0) / (np.sqrt(2.0 * l - 3.0) * l)
        for m in range(1, l - 1):
            k += 1
            f1[k] = np.sqrt(2.0 * l + 1.0) * np.sqrt(2.0 * l - 1.0) / (np.sqrt(l + m) * np.sqrt(l - m))
            f2[k] = (
                np.sqrt(2.0 * l + 1.0) * np.sqrt(l - m - 1.0) * np.sqrt(l + m - 1.0) /
                (np.sqrt(2.0 * l - 3.0) * np.sqrt(l + m) * np.sqrt(l - m))
            )
        k += 2

    for i in range(n_points):
        x = np.cos(phi_bar[i])
        u = np.sqrt(1.0 - x * x)
        if u == 0.0:
            u = np.finfo(np.float64).eps

        p = np.zeros(tri_size, dtype=np.float64)
        p[0] = 1.0
        if nmax >= 1:
            p[1] = np.sqrt(3.0) * x

        k = 1
        for l in range(2, nmax + 1):
            k += l
            p[k] = f1[k] * x * p[k - l] - f2[k] * p[k - 2 * l + 1]

        pmm = np.sqrt(2.0) * scalef
        rescalem = 1.0 / scalef
        kstart = 0

        for m in range(1, nmax):
            rescalem *= u
            kstart += m + 1
            pmm = pmm * np.sqrt(2.0 * m + 1.0) / np.sqrt(2.0 * m)
            p[kstart] = pmm

            k = kstart + m + 1
            p[k] = x * np.sqrt(2.0 * m + 3.0) * pmm

            for l in range(m + 2, nmax + 1):
                k += l
                p[k] = x * f1[k] * p[k - l] - f2[k] * p[k - 2 * l + 1]
                p[k - 2 * l + 1] *= rescalem

            p[k] *= rescalem
            p[k - nmax] *= rescalem

        if nmax > 0:
            rescalem *= u
            kstart += nmax + 1
            p[kstart] = pmm * np.sqrt(2.0 * nmax + 1.0) / np.sqrt(2.0 * nmax) * rescalem

        for m in range(nmax + 1):
            for l in range(m, nmax + 1):
                lm = (l * (l + 1)) // 2 + m
                plm[i, l, m] = p[lm]
                if compute_derivative:
                    if l == m:
                        dplm[i, l, m] = m * (x / u) * plm[i, l, m]
                    else:
                        flm = np.sqrt(((l * l - m * m) * (2.0 * l + 1.0)) / (2.0 * l - 1.0))
                        dplm[i, l, m] = (1.0 / u) * (l * x * plm[i, l, m] - flm * plm[i, l - 1, m])

    return plm, dplm


@njit(parallel=True)
def compute_alfs_holmes_pointwise_parallel(phi_bar, nmax, compute_derivative):
    '''
    Compute Holmes-Featherstone ALFs point-by-point using threaded Numba.
    '''
    n_points = len(phi_bar)
    tri_size = (nmax + 1) * (nmax + 2) // 2
    scalef = 1.0e-280

    f1 = np.zeros(tri_size, dtype=np.float64)
    f2 = np.zeros(tri_size, dtype=np.float64)
    plm = np.zeros((n_points, nmax + 1, nmax + 1), dtype=np.float64)
    dplm = np.zeros((n_points, nmax + 1, nmax + 1), dtype=np.float64) if compute_derivative else np.zeros((1, 1, 1), dtype=np.float64)

    k = 2
    for l in range(2, nmax + 1):
        k += 1
        f1[k] = np.sqrt(2.0 * l - 1.0) * np.sqrt(2.0 * l + 1.0) / l
        f2[k] = (l - 1.0) * np.sqrt(2.0 * l + 1.0) / (np.sqrt(2.0 * l - 3.0) * l)
        for m in range(1, l - 1):
            k += 1
            f1[k] = np.sqrt(2.0 * l + 1.0) * np.sqrt(2.0 * l - 1.0) / (np.sqrt(l + m) * np.sqrt(l - m))
            f2[k] = (
                np.sqrt(2.0 * l + 1.0) * np.sqrt(l - m - 1.0) * np.sqrt(l + m - 1.0) /
                (np.sqrt(2.0 * l - 3.0) * np.sqrt(l + m) * np.sqrt(l - m))
            )
        k += 2

    for i in prange(n_points):
        x = np.cos(phi_bar[i])
        u = np.sqrt(1.0 - x * x)
        if u == 0.0:
            u = np.finfo(np.float64).eps

        p = np.zeros(tri_size, dtype=np.float64)
        p[0] = 1.0
        if nmax >= 1:
            p[1] = np.sqrt(3.0) * x

        k = 1
        for l in range(2, nmax + 1):
            k += l
            p[k] = f1[k] * x * p[k - l] - f2[k] * p[k - 2 * l + 1]

        pmm = np.sqrt(2.0) * scalef
        rescalem = 1.0 / scalef
        kstart = 0

        for m in range(1, nmax):
            rescalem *= u
            kstart += m + 1
            pmm = pmm * np.sqrt(2.0 * m + 1.0) / np.sqrt(2.0 * m)
            p[kstart] = pmm

            k = kstart + m + 1
            p[k] = x * np.sqrt(2.0 * m + 3.0) * pmm

            for l in range(m + 2, nmax + 1):
                k += l
                p[k] = x * f1[k] * p[k - l] - f2[k] * p[k - 2 * l + 1]
                p[k - 2 * l + 1] *= rescalem

            p[k] *= rescalem
            p[k - nmax] *= rescalem

        if nmax > 0:
            rescalem *= u
            kstart += nmax + 1
            p[kstart] = pmm * np.sqrt(2.0 * nmax + 1.0) / np.sqrt(2.0 * nmax) * rescalem

        for m in range(nmax + 1):
            for l in range(m, nmax + 1):
                lm = (l * (l + 1)) // 2 + m
                plm[i, l, m] = p[lm]
                if compute_derivative:
                    if l == m:
                        dplm[i, l, m] = m * (x / u) * plm[i, l, m]
                    else:
                        flm = np.sqrt(((l * l - m * m) * (2.0 * l + 1.0)) / (2.0 * l - 1.0))
                        dplm[i, l, m] = (1.0 / u) * (l * x * plm[i, l, m] - flm * plm[i, l - 1, m])

    return plm, dplm


@njit(parallel=True)
def compute_alfs_holmes_numba(phi_bar, nmax):
    '''
    Compute fully normalized ALFs using the Holmes-Featherstone scaled recursion.
    '''
    x = np.cos(phi_bar)
    jm = len(x)
    scalef = 1.0e-280

    tri_size = (nmax + 1) * (nmax + 2) // 2
    f1 = np.zeros(tri_size, dtype=np.float64)
    f2 = np.zeros(tri_size, dtype=np.float64)
    p = np.zeros((tri_size, jm), dtype=np.float64)
    plm = np.zeros((jm, nmax + 1, nmax + 1), dtype=np.float64)
    k = 2
    for l in range(2, nmax + 1):
        k += 1
        f1[k] = np.sqrt(2.0 * l - 1.0) * np.sqrt(2.0 * l + 1.0) / l
        f2[k] = (l - 1.0) * np.sqrt(2.0 * l + 1.0) / (np.sqrt(2.0 * l - 3.0) * l)
        for m in range(1, l - 1):
            k += 1
            f1[k] = np.sqrt(2.0 * l + 1.0) * np.sqrt(2.0 * l - 1.0) / (np.sqrt(l + m) * np.sqrt(l - m))
            f2[k] = (
                np.sqrt(2.0 * l + 1.0) * np.sqrt(l - m - 1.0) * np.sqrt(l + m - 1.0) /
                (np.sqrt(2.0 * l - 3.0) * np.sqrt(l + m) * np.sqrt(l - m))
            )
        k += 2

    u = np.sqrt(1.0 - x * x)
    for j in prange(jm):
        if u[j] == 0.0:
            u[j] = np.finfo(np.float64).eps
        p[0, j] = 1.0
        p[1, j] = np.sqrt(3.0) * x[j]

    k = 1
    for l in range(2, nmax + 1):
        k += l
        for j in prange(jm):
            p[k, j] = f1[k] * x[j] * p[k - l, j] - f2[k] * p[k - 2 * l + 1, j]

    pmm = np.sqrt(2.0) * scalef
    rescalem = np.full(jm, 1.0 / scalef, dtype=np.float64)
    kstart = 0

    for m in range(1, nmax):
        for j in prange(jm):
            rescalem[j] = rescalem[j] * u[j]
        kstart += m + 1
        pmm = pmm * np.sqrt(2.0 * m + 1.0) / np.sqrt(2.0 * m)
        for j in prange(jm):
            p[kstart, j] = pmm
        k = kstart + m + 1
        for j in prange(jm):
            p[k, j] = x[j] * np.sqrt(2.0 * m + 3.0) * pmm
        for l in range(m + 2, nmax + 1):
            k += l
            for j in prange(jm):
                p[k, j] = x[j] * f1[k] * p[k - l, j] - f2[k] * p[k - 2 * l + 1, j]
                p[k - 2 * l + 1, j] = p[k - 2 * l + 1, j] * rescalem[j]
        for j in prange(jm):
            p[k, j] = p[k, j] * rescalem[j]
            p[k - nmax, j] = p[k - nmax, j] * rescalem[j]

    if nmax > 0:
        for j in prange(jm):
            rescalem[j] = rescalem[j] * u[j]
        kstart += nmax + 1
        for j in prange(jm):
            p[kstart, j] = pmm * np.sqrt(2.0 * nmax + 1.0) / np.sqrt(2.0 * nmax) * rescalem[j]

    for m in range(nmax + 1):
        for l in range(m, nmax + 1):
            lm = (l * (l + 1)) // 2 + m
            for j in prange(jm):
                plm[j, l, m] = p[lm, j]

    return plm


@njit(parallel=True)
def compute_alfs_holmes_with_deriv_numba(phi_bar, nmax):
    '''
    Compute fully normalized ALFs and first derivatives using the
    Holmes-Featherstone scaled recursion.
    '''
    x = np.cos(phi_bar)
    jm = len(x)
    scalef = 1.0e-280

    tri_size = (nmax + 1) * (nmax + 2) // 2
    f1 = np.zeros(tri_size, dtype=np.float64)
    f2 = np.zeros(tri_size, dtype=np.float64)
    p = np.zeros((tri_size, jm), dtype=np.float64)
    plm = np.zeros((jm, nmax + 1, nmax + 1), dtype=np.float64)
    dplm = np.zeros((jm, nmax + 1, nmax + 1), dtype=np.float64)

    k = 2
    for l in range(2, nmax + 1):
        k += 1
        f1[k] = np.sqrt(2.0 * l - 1.0) * np.sqrt(2.0 * l + 1.0) / l
        f2[k] = (l - 1.0) * np.sqrt(2.0 * l + 1.0) / (np.sqrt(2.0 * l - 3.0) * l)
        for m in range(1, l - 1):
            k += 1
            f1[k] = np.sqrt(2.0 * l + 1.0) * np.sqrt(2.0 * l - 1.0) / (np.sqrt(l + m) * np.sqrt(l - m))
            f2[k] = (
                np.sqrt(2.0 * l + 1.0) * np.sqrt(l - m - 1.0) * np.sqrt(l + m - 1.0) /
                (np.sqrt(2.0 * l - 3.0) * np.sqrt(l + m) * np.sqrt(l - m))
            )
        k += 2

    u = np.sqrt(1.0 - x * x)
    for j in prange(jm):
        if u[j] == 0.0:
            u[j] = np.finfo(np.float64).eps
        p[0, j] = 1.0
        p[1, j] = np.sqrt(3.0) * x[j]

    k = 1
    for l in range(2, nmax + 1):
        k += l
        for j in prange(jm):
            p[k, j] = f1[k] * x[j] * p[k - l, j] - f2[k] * p[k - 2 * l + 1, j]

    pmm = np.sqrt(2.0) * scalef
    rescalem = np.full(jm, 1.0 / scalef, dtype=np.float64)
    kstart = 0

    for m in range(1, nmax):
        for j in prange(jm):
            rescalem[j] = rescalem[j] * u[j]
        kstart += m + 1
        pmm = pmm * np.sqrt(2.0 * m + 1.0) / np.sqrt(2.0 * m)
        for j in prange(jm):
            p[kstart, j] = pmm
        k = kstart + m + 1
        for j in prange(jm):
            p[k, j] = x[j] * np.sqrt(2.0 * m + 3.0) * pmm
        for l in range(m + 2, nmax + 1):
            k += l
            for j in prange(jm):
                p[k, j] = x[j] * f1[k] * p[k - l, j] - f2[k] * p[k - 2 * l + 1, j]
                p[k - 2 * l + 1, j] = p[k - 2 * l + 1, j] * rescalem[j]
        for j in prange(jm):
            p[k, j] = p[k, j] * rescalem[j]
            p[k - nmax, j] = p[k - nmax, j] * rescalem[j]

    if nmax > 0:
        for j in prange(jm):
            rescalem[j] = rescalem[j] * u[j]
        kstart += nmax + 1
        for j in prange(jm):
            p[kstart, j] = pmm * np.sqrt(2.0 * nmax + 1.0) / np.sqrt(2.0 * nmax) * rescalem[j]

    for m in range(nmax + 1):
        for l in range(m, nmax + 1):
            lm = (l * (l + 1)) // 2 + m
            flm = 0.0
            if l != m:
                flm = np.sqrt(((l * l - m * m) * (2.0 * l + 1.0)) / (2.0 * l - 1.0))
            for j in prange(jm):
                plm[j, l, m] = p[lm, j]
                if l == m:
                    dplm[j, l, m] = m * (x[j] / u[j]) * plm[j, l, m]
                else:
                    dplm[j, l, m] = (1.0 / u[j]) * (l * x[j] * plm[j, l, m] - flm * plm[j, l - 1, m])

    return plm, dplm
