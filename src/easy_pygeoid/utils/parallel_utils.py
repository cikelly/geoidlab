import numpy as np

def compute_tc_row(
    i, ncols_P, dm, lamp, 
    phip, Hp, ori_topo, 
    X, Y, Z, Xp, Yp, Zp, 
    radius, G, rho, dx, dy
) -> np.ndarray:
    '''
    Compute a single row of the terrain correction matrix.
    
    Parameters
    ----------
    i         : row index
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
    tc_row    : 1D array of terrain correction values
    '''
    tc_row = np.zeros(ncols_P)
    m1 = 1
    m2 = dm
    
    coslamp = np.cos(lamp[i, :])
    sinlamp = np.sin(lamp[i, :])
    cosphip = np.cos(phip[i, :])
    sinphip = np.sin(phip[i, :])
    
    for j in range(ncols_P):
        smallH = ori_topo['z'].values[i:i+dm, m1:m2]
        smallX = X[i:i+dm, m1:m2]
        smallY = Y[i:i+dm, m1:m2]
        smallZ = Z[i:i+dm, m1:m2]

        # Local coordinates (x, y)
        x = coslamp[j] * (smallY - Yp[i, j]) - \
            sinlamp[j] * (smallX - Xp[i, j])
        y = cosphip[j] * (smallZ - Zp[i, j]) - \
            coslamp[j] * sinphip[j] * (smallX - Xp[i, j]) - \
            sinlamp[j] * sinphip[j] * (smallY - Yp[i, j])

        # Distances
        d = np.hypot(x, y)
        d = np.where(d <= radius, d, np.nan)
        d3 = d * d * d
        d5 = d3 * d * d
        d7 = d5 * d * d
        
        # Integrate the terrain correction
        DH2 = (smallH - Hp[i, j]) * (smallH - Hp[i, j])
        DH4 = DH2 * DH2
        c1  = 1/2 * G * rho * dx * dy * np.nansum(DH2 / d3)
        c2  = -3/8 * G * rho * dx * dy * np.nansum(DH4 / d5)
        c3  = 5/16 * G * rho * dx * dy * np.nansum(DH2 * DH2 * DH2 / d7)
        tc_row[j] = (c1 + c2 + c3) * 1e5  # [mGal]
        
        # Moving window
        m1 += 1
        m2 += 1
    
    return i, tc_row