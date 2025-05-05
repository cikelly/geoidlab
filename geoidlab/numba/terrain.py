############################################################
# Utilities for numba optimized terrain quantities         #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################
import numpy as np
from numba import njit

@njit
def compute_tc_chunk(
    row_start: int, row_end: int, ncols_P: int, coslamp: np.ndarray, sinlamp: np.ndarray,
    cosphip: np.ndarray, sinphip: np.ndarray, Hp: np.ndarray, ori_topo: np.ndarray,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, Xp: np.ndarray, Yp: np.ndarray,
    Zp: np.ndarray, radius: float, G_rho_dxdy: float, window_indices: np.ndarray
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

    for i in range(row_start, row_end):
        # m1 = 0
        # m2 = dm
        
        coslamp_i = coslamp[i, :]
        sinlamp_i = sinlamp[i, :]
        cosphip_i = cosphip[i, :]
        sinphip_i = sinphip[i, :]

        for j in range(ncols_P):
            i_start, i_end, j_start, j_end = window_indices[i, j]
            smallH = ori_topo[i_start:i_end, j_start:j_end]
            smallX = X[i_start:i_end, j_start:j_end]
            smallY = Y[i_start:i_end, j_start:j_end]
            smallZ = Z[i_start:i_end, j_start:j_end]

            # Local coordinates (x, y)
            x = coslamp_i[j] * (smallY - Yp[i, j]) - \
                sinlamp_i[j] * (smallX - Xp[i, j])
            y = cosphip_i[j] * (smallZ - Zp[i, j]) - \
                coslamp_i[j] * sinphip_i[j] * (smallX - Xp[i, j]) - \
                sinlamp_i[j] * sinphip_i[j] * (smallY - Yp[i, j])

            # Distances
            d = np.hypot(x, y)
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

    return row_start, row_end, tc_chunk

@njit
def compute_tc_chunk_cap(
    row_start: int, row_end: int, ncols_P: int, coslamp: np.ndarray, sinlamp: np.ndarray,
    cosphip: np.ndarray, sinphip: np.ndarray, Hp: np.ndarray, Xp: np.ndarray, Yp: np.ndarray,
    Zp: np.ndarray, radius: float, G_rho_dxdy: float, window_data: np.ndarray
) -> tuple[int, int, np.ndarray]:
    '''
    Compute a chunk of rows for the terrain correction matrix in cap mode.

    Parameters
    ----------
    row_start  : Starting row index (inclusive).
    row_end    : Ending row index (exclusive).
    ncols_P    : Number of columns in the sub-grid.
    coslamp, 
    sinlamp, 
    cosphip, 
    sinphip    : Trigonometric arrays for coordinate transformation.
    Hp         : Height of computation points (sub-grid topography).
    Xp, 
    Yp, 
    Zp         : Cartesian coordinates of the sub-grid.
    radius     : Integration radius [m].
    G_rho_dxdy : Precomputed constant (G * rho * dx * dy).
    window_data: Array of shape (row_end-row_start, ncols_P, 4, max_rows, max_cols) containing
                [win_H, win_X, win_Y, win_Z] for each computation point, pre-masked with NaNs.

    Returns
    -------
    row_start : Starting row index.
    row_end   : Ending row index.
    tc_chunk  : 2D array of terrain correction values for the chunk [mGal].
    '''
    tc_chunk = np.zeros((row_end - row_start, ncols_P))

    for i in range(row_start, row_end):
        coslamp_i = coslamp[i, :]
        sinlamp_i = sinlamp[i, :]
        cosphip_i = cosphip[i, :]
        sinphip_i = sinphip[i, :]

        for j in range(ncols_P):
            # Extract pre-masked windowed data
            win_H = window_data[i - row_start, j, 0, :, :]
            win_X = window_data[i - row_start, j, 1, :, :]
            win_Y = window_data[i - row_start, j, 2, :, :]
            win_Z = window_data[i - row_start, j, 3, :, :]

            # Local coordinates (x, y)
            x = coslamp_i[j] * (win_Y - Xp[i, j]) - sinlamp_i[j] * (win_X - Xp[i, j])
            y = cosphip_i[j] * (win_Z - Zp[i, j]) - \
                coslamp_i[j] * sinphip_i[j] * (win_X - Xp[i, j]) - \
                sinlamp_i[j] * sinphip_i[j] * (win_Y - Yp[i, j])

            # Distances
            d = np.hypot(x, y)
            # Mask computation point (already masked by get_cap_window, but ensure safety)
            for k in range(d.shape[0]):
                for l in range(d.shape[1]):
                    if d[k, l] == 0:
                        d[k, l] = np.nan

            d3 = d * d * d
            d5 = d3 * d * d
            d7 = d5 * d * d

            # Integrate the terrain correction
            DH2 = (win_H - Hp[i, j]) ** 2
            DH4 = DH2 * DH2
            DH6 = DH4 * DH2

            c1 = 0.5 * G_rho_dxdy * np.nansum(DH2 / d3)      # 1/2
            c2 = -0.375 * G_rho_dxdy * np.nansum(DH4 / d5)   # 3/8
            c3 = 0.3125 * G_rho_dxdy * np.nansum(DH6 / d7)   # 5/16
            tc_chunk[i - row_start, j] = (c1 + c2 + c3) * 1e5  # [mGal]

    return row_start, row_end, tc_chunk

@njit
def compute_rtm_tc_chunk(
    row_start: int, row_end: int, ncols_P: int, coslamp: np.ndarray, sinlamp: np.ndarray, 
    cosphip: np.ndarray, sinphip: np.ndarray, Hp: np.ndarray, ori_topo: np.ndarray, X: np.ndarray, Y: np.ndarray, 
    Z: np.ndarray, Xp: np.ndarray, Yp: np.ndarray, Zp: np.ndarray, radius: float, G_rho_dxdy: float,
    Hp_ref: np.ndarray, ref_topo: np.ndarray, window_indices: np.ndarray
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
    tc_chunk  : 2D array of RTM terrain correction values for the chunk
    '''
    tc_chunk = np.zeros((row_end - row_start, ncols_P))

    for i in range(row_start, row_end):
        coslamp_i = coslamp[i, :]
        sinlamp_i = sinlamp[i, :]
        cosphip_i = cosphip[i, :]
        sinphip_i = sinphip[i, :]

        for j in range(ncols_P):
            i_start, i_end, j_start, j_end = window_indices[i, j]
            smallH = ori_topo[i_start:i_end, j_start:j_end]
            smallH_ref = ref_topo[i_start:i_end, j_start:j_end]
            smallX = X[i_start:i_end, j_start:j_end]
            smallY = Y[i_start:i_end, j_start:j_end]
            smallZ = Z[i_start:i_end, j_start:j_end]

            # Local coordinates (x, y)
            x = coslamp_i[j] * (smallY - Yp[i, j]) - \
                sinlamp_i[j] * (smallX - Xp[i, j])
            y = cosphip_i[j] * (smallZ - Zp[i, j]) - \
                coslamp_i[j] * sinphip_i[j] * (smallX - Xp[i, j]) - \
                sinlamp_i[j] * sinphip_i[j] * (smallY - Yp[i, j])

            # Distances
            d = np.hypot(x, y)

            # Numba compliant masking
            for k in range(d.shape[0]):
                for l in range(d.shape[1]):
                    if d[k, l] > radius or d[k, l] == 0:
                        d[k, l] = np.nan

            d3 = d * d * d
            d5 = d3 * d * d
            d7 = d5 * d * d
            
            # Powers of height differences
            DH2    = (smallH - Hp[i, j]) ** 2
            DH4    = DH2 * DH2
            DH6    = DH4 * DH2
            
            DH_ref2 = (smallH_ref - Hp_ref[i, j]) ** 2
            DH_ref4 = DH_ref2 * DH_ref2
            DH_ref6 = DH_ref4 * DH_ref2
            
            # Integrate the RMT terrain correction

            c1  = 0.5 *  G_rho_dxdy * np.nansum((DH_ref2 - DH2) / d3)      # 1/2
            c2  = -0.375 * G_rho_dxdy * np.nansum((DH_ref4 - DH4) / d5)    # 3/8
            c3  = 0.3125 * G_rho_dxdy * np.nansum((DH_ref6 - DH6) / d7)    # 5/16
            tc_chunk[i - row_start, j] = (c1 + c2 + c3) * 1e5  # [mGal]

    return row_start, row_end, tc_chunk


@njit
def compute_ind_chunk(
    row_start: int, row_end: int, ncols_P: int, coslamp: np.ndarray, sinlamp: np.ndarray, 
    cosphip: np.ndarray, sinphip: np.ndarray, Hp: np.ndarray, ori_topo: np.ndarray, X: np.ndarray, Y: np.ndarray, 
    Z: np.ndarray, Xp: np.ndarray, Yp: np.ndarray, Zp: np.ndarray, radius: float, G_rho_dxdy: float,
    Hp_ref: np.ndarray, ref_topo: np.ndarray, window_indices: np.ndarray
) -> tuple[int, int, np.ndarray]:
    '''
    Compute a chunk of rows for the indirect effect of Helmert's second method of condensation

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
    ind_chunk : 2D array of terrain correction values for the chunk
    '''
    ind_chunk = np.zeros((row_end - row_start, ncols_P))

    for i in range(row_start, row_end):
        
        coslamp_i = coslamp[i, :]
        sinlamp_i = sinlamp[i, :]
        cosphip_i = cosphip[i, :]
        sinphip_i = sinphip[i, :]

        for j in range(ncols_P):
            # smallH = ori_topo['z'].values[i:i+dn, m1:m2]
            i_start, i_end, j_start, j_end = window_indices[i, j]
            smallH = ori_topo[i_start:i_end, j_start:j_end]
            smallX = X[i_start:i_end, j_start:j_end]
            smallY = Y[i_start:i_end, j_start:j_end]
            smallZ = Z[i_start:i_end, j_start:j_end]

            # Local coordinates (x, y)
            x = coslamp_i[j] * (smallY - Yp[i, j]) - \
                sinlamp_i[j] * (smallX - Xp[i, j])
            y = cosphip_i[j] * (smallZ - Zp[i, j]) - \
                coslamp_i[j] * sinphip_i[j] * (smallX - Xp[i, j]) - \
                sinlamp_i[j] * sinphip_i[j] * (smallY - Yp[i, j])

            # Distances
            d = np.hypot(x, y)

            # Numba compliant masking
            for k in range(d.shape[0]):
                for l in range(d.shape[1]):
                    if d[k, l] > radius or d[k, l] == 0:
                        d[k, l] = np.nan

            d3 = d * d * d
            d5 = d3 * d * d
            d7 = d5 * d * d

            # Potential change of the irregular part of topography
            # Powers of height
            Hp3 = Hp[i, j] ** 3
            Hp5 = Hp3 * Hp[i, j] * Hp[i, j]
            Hp7 = Hp5 * Hp[i, j] * Hp[i, j]
            H3  = smallH ** 3
            H5  = H3 * smallH * smallH
            H7  = H5 * smallH * smallH

            v2  = -1/6 * np.nansum((H3 - Hp3) / d3) 
            v3  = 0.075 * np.nansum((H5 - Hp5) / d5)    # 3/40
            v4  = -15/336 * np.nansum((H7 - Hp7) / d7)  
            ind_chunk[i - row_start, j] = G_rho_dxdy * (v2 + v3 + v4)

    return row_start, row_end, ind_chunk

@njit
def compute_rtm_height_anomaly_chunk(
    row_start: int, row_end: int, ncols_P: int, coslamp: np.ndarray, sinlamp: np.ndarray,
    cosphip: np.ndarray, sinphip: np.ndarray, Hp: np.ndarray, ori_topo: np.ndarray,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, Xp: np.ndarray, Yp: np.ndarray,
    Zp: np.ndarray, radius: float, G_rho_dxdy: float, HrefP: np.ndarray, ref_topo: np.ndarray,
    window_indices: np.ndarray
) -> tuple[int, int, np.ndarray]:
    '''
    Compute a chunk of rows for the RTM height anomaly.

    Parameters
    ----------
    row_start : Starting row index (inclusive)
    row_end   : Ending row index (exclusive)
    ncols_P   : Number of columns in the sub-grid
    coslamp, sinlamp, cosphip, sinphip : Trigonometric functions of computation points
    Hp        : Height of computation points (original topography)
    ori_topo  : Original topography
    X, Y, Z   : Cartesian coordinates of the original topography
    Xp, Yp, Zp: Cartesian coordinates of the sub-grid
    radius    : Integration radius [m]
    G_rho_dxdy: G * rho * dx * dy
    HrefP     : Height of computation points (reference topography)
    ref_topo  : Reference topography
    window_indices : Precomputed window indices for each computation point

    Returns
    -------
    row_start : Starting row index
    row_end   : Ending row index
    z_rtm_chunk : 2D array of RTM height anomaly values for the chunk [m]
    '''
    z_rtm_chunk = np.zeros((row_end - row_start, ncols_P))

    for i in range(row_start, row_end):
        coslamp_i = coslamp[i, :]
        sinlamp_i = sinlamp[i, :]
        cosphip_i = cosphip[i, :]
        sinphip_i = sinphip[i, :]

        for j in range(ncols_P):
            i_start, i_end, j_start, j_end = window_indices[i, j]
            smallH = ori_topo[i_start:i_end, j_start:j_end]
            smallH_ref = ref_topo[i_start:i_end, j_start:j_end]
            smallX = X[i_start:i_end, j_start:j_end]
            smallY = Y[i_start:i_end, j_start:j_end]
            smallZ = Z[i_start:i_end, j_start:j_end]

            # Local coordinates (x, y)
            x = coslamp_i[j] * (smallY - Yp[i, j]) - \
                sinlamp_i[j] * (smallX - Xp[i, j])
            y = cosphip_i[j] * (smallZ - Zp[i, j]) - \
                coslamp_i[j] * sinphip_i[j] * (smallX - Xp[i, j]) - \
                sinlamp_i[j] * sinphip_i[j] * (smallY - Yp[i, j])

            # Distances
            d = np.hypot(x, y)
            for k in range(d.shape[0]):
                for l in range(d.shape[1]):
                    if d[k, l] > radius or d[k, l] == 0:
                        d[k, l] = np.nan
            d3 = d * d * d
            d5 = d3 * d * d

            # Height differences
            z1 = smallH - smallH_ref
            z3 = smallH**3 - smallH_ref**3
            z5 = smallH**5 - smallH_ref**5

            # Integrate the RTM height anomaly
            c1 = np.nansum(z1 / d)
            c2 = -1/6 * np.nansum(z3 / d3)
            c3 = 0.075 * np.nansum(z5 / d5)
            z_rtm_chunk[i - row_start, j] = (1 / 9.82) * G_rho_dxdy * (c1 + c2 + c3)

    return row_start, row_end, z_rtm_chunk