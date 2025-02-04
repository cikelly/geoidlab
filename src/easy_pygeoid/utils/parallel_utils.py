############################################################
# Utilities for modeling terrain quantities                #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################
import numpy as np
import bottleneck as bn
from numba import njit

from numpy.lib.stride_tricks import sliding_window_view

@njit
def compute_tc_chunk(
    row_start: int, row_end: int, ncols_P: int, dm: int, dn: int, coslamp: np.ndarray, sinlamp: np.ndarray, 
    cosphip: np.ndarray, sinphip: np.ndarray, Hp: np.ndarray, ori_topo: np.ndarray, X: np.ndarray, Y: np.ndarray, 
    Z: np.ndarray, Xp: np.ndarray, Yp: np.ndarray, Zp: np.ndarray, radius: float, G_rho_dxdy: float
) -> tuple[int, int, np.ndarray]:
    '''
    Compute a chunk of rows for the terrain correction matrix.
    
    Parameters
    ----------
    row_start : starting row index (inclusive)
    row_end   : ending row index (exclusive)
    ncols_P   : number of columns in the sub-grid
    dm        : number of rows in the moving window
    lamp      : longitude of the computation points
    phip      : latitude of the computation points
    Hp        : height of the computation points
    ori_topo  : original topography
    X, Y, Z   : cartesian coordinates of the original topography
    Xp, Yp, Zp: cartesian coordinates of the sub-grid
    radius    : integration radius [km]
    G         : gravitational constant
    rho       : density of the Earth
    dx, dy    : grid size in x and y directions
    
    Returns
    -------
    row_start : starting row index
    row_end   : ending row index
    tc_chunk  : 2D array of terrain correction values for the chunk
    '''
    tc_chunk = np.zeros((row_end - row_start, ncols_P))
    
    # Create sliding window views for the arrays
    ## H_view = sliding_window_view(ori_topo['z'].values, (dn, dm))
    # H_view = sliding_window_view(ori_topo, (dn, dm))
    # X_view = sliding_window_view(X, (dn, dm))
    # Y_view = sliding_window_view(Y, (dn, dm))
    # Z_view = sliding_window_view(Z, (dn, dm))
    
    for i in range(row_start, row_end):
        m1 = 1
        m2 = dm
        
        coslamp_i = coslamp[i, :]
        sinlamp_i = sinlamp[i, :]
        cosphip_i = cosphip[i, :]
        sinphip_i = sinphip[i, :]
        
        for j in range(ncols_P):
            # smallH = ori_topo['z'].values[i:i+dn, m1:m2]
            smallH = ori_topo[i:i+dn, m1:m2]
            smallX = X[i:i+dn, m1:m2]
            smallY = Y[i:i+dn, m1:m2]
            smallZ = Z[i:i+dn, m1:m2]
            
            # Extract subarrays using sliding window views
            # smallH = H_view[i, j]
            # smallX = X_view[i, j]
            # smallY = Y_view[i, j]
            # smallZ = Z_view[i, j]

            # Local coordinates (x, y)
            x = coslamp_i[j] * (smallY - Yp[i, j]) - \
                sinlamp_i[j] * (smallX - Xp[i, j])
            y = cosphip_i[j] * (smallZ - Zp[i, j]) - \
                coslamp_i[j] * sinphip_i[j] * (smallX - Xp[i, j]) - \
                sinlamp_i[j] * sinphip_i[j] * (smallY - Yp[i, j])

            # Distances
            d = np.hypot(x, y)
            # d[d > radius] = np.nan
            # d[d == 0] = np.nan
            # Numba compliant masking
            for k in range(d.shape[0]):
                for l in range(d.shape[1]):
                    if d[k, l] > radius or d[k, l] == 0:
                        d[k, l] = np.nan
            
            d3 = d * d * d
            d5 = d3 * d * d
            d7 = d5 * d * d
            
            # Integrate the terrain correction
            DH2 = (smallH - Hp[i, j]) ** 2 
            DH4 = DH2 * DH2
            DH6 = DH4 * DH2
            
            c1  = 0.5 *  G_rho_dxdy * np.nansum(DH2 / d3)      # 1/2
            c2  = -0.375 * G_rho_dxdy * np.nansum(DH4 / d5)    # 3/8
            c3  = 0.3125 * G_rho_dxdy * np.nansum(DH6 / d7)    # 5/16
            tc_chunk[i - row_start, j] = (c1 + c2 + c3) * 1e5  # [mGal]
            
            # Moving window
            m1 += 1
            m2 += 1
    
    return row_start, row_end, tc_chunk