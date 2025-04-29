############################################################
# Utilities for numba optimized stokes kernel              #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import numpy as np
from numba import njit
@njit
def stokes_kernel(lonp, latp, lon, lat) -> tuple:
    '''Compute Stokes' kernel for vectorized inputs.'''
    # Ensure lonp, latp are broadcastable
    lonp = lonp[:, np.newaxis]
    latp = latp[:, np.newaxis]
    
    cos_dlam = np.cos(lon) * np.cos(lonp) + np.sin(lon) * np.sin(lonp)
    cos_psi = np.sin(latp) * np.sin(lat) + np.cos(latp) * np.cos(lat) * cos_dlam
    sin2_psi_2 = np.sin((latp - lat)/2)**2 + np.cos(latp) * np.cos(lat) * np.sin((lonp - lon)/2)**2
    
    log_arg = np.sqrt(sin2_psi_2) + sin2_psi_2
    S = np.where(
        log_arg <= 0,
        0.0,
        1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - 3 * cos_psi * np.log(log_arg)
    )
    return S, cos_psi