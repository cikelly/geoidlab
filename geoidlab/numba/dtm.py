############################################################
# Utilities for numba optimized DTM modeling               #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_harmonic_sum(Pnm, HCnm, HSnm, cosm, sinm) -> np.ndarray:
    num_points = Pnm.shape[0]
    nmax_plus_1 = Pnm.shape[1]
    H = np.zeros(num_points)
    for i in prange(num_points):
        total = 0.0
        for n in range(nmax_plus_1):
            for m in range(n + 1):
                cos_term = HCnm[n, m] * Pnm[i, n, m] * cosm[m, i]
                sin_term = HSnm[n, m] * Pnm[i, n, m] * sinm[m, i]
                total += cos_term + sin_term
        H[i] = total
    return H

@njit(parallel=True)
def compute_harmonic_sum_precomputed(HC_Pnm, HS_Pnm, cosm, sinm) -> np.ndarray:
    num_points = HC_Pnm.shape[0]
    nmax_plus_1 = HC_Pnm.shape[1]
    H = np.zeros(num_points)
    for i in prange(num_points):
        total = 0.0
        for n in range(nmax_plus_1):
            for m in range(n + 1):
                total += HC_Pnm[i, n, m] * cosm[m, i] + HS_Pnm[i, n, m] * sinm[m, i]
        H[i] = total
    return H