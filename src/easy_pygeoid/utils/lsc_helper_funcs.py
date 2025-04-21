import numpy as np
from typing import Union, Tuple, List, Optional

def compute_spatial_covariance(
    X: np.ndarray, 
    Y: np.ndarray, 
    G: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute empirical covariance of 2D spatial data.
    
    Parameters
    ----------
    X         : X coordinates of observations
    Y         : Y coordinates of observations
    G         : Observation values
    
    Returns
    -------
    covariance: Empirical covariance values
    covdist   : Corresponding distances
    '''
    smax = np.sqrt((X.max() - X.min())**2 + (Y.max() - Y.min())**2)
    ds = np.sqrt((2 * np.pi * (smax / 2)**2) / len(X))
    n_bins = int(np.round(smax / ds)) + 2
    covariance = np.zeros(n_bins)
    ncov = np.zeros(n_bins)
    
    # Vectorize distance calculations when possible
    for i in range(len(G)):
        dx = X[i] - X
        dy = Y[i] - Y
        r = np.sqrt(dx**2 + dy**2)
        
        # Skip self-distance
        mask = (r > 0) & (r < smax)
        ir = np.round(r[mask] / ds).astype(int)
        
        # Only process bins that are within range
        valid_bins = ir < n_bins
        if np.any(valid_bins):
            np.add.at(covariance, ir[valid_bins], G[i] * G[mask][valid_bins])
            np.add.at(ncov, ir[valid_bins], 1)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-12
    covariance = np.where(ncov > 0, covariance / (ncov + epsilon), 0)
    covdist = np.arange(n_bins) * ds
    return covariance, covdist


def compute_spatial_covariance_robust(
    X: np.ndarray, 
    Y: np.ndarray, 
    G: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute robust (median-based) empirical covariance of 2D spatial data.
    Parameters
    ----------
    X         : X coordinates of observations
    Y         : Y coordinates of observations
    G         : Observation values
    Returns
    -------
    covariance: Empirical covariance values (median-based)
    covdist   : Corresponding distances
    '''
    smax = np.sqrt((X.max() - X.min())**2 + (Y.max() - Y.min())**2)
    ds = np.sqrt((2 * np.pi * (smax / 2)**2) / len(X))
    n_bins = int(np.round(smax / ds)) + 2
    covariance = np.zeros(n_bins)
    ncov = np.zeros(n_bins)
    for i in range(len(G)):
        dx = X[i] - X
        dy = Y[i] - Y
        r = np.sqrt(dx**2 + dy**2)
        mask = (r > 0) & (r < smax)
        ir = np.round(r[mask] / ds).astype(int)
        valid_bins = ir < n_bins
        if np.any(valid_bins):
            # Use median instead of mean for robust estimation
            for bin_idx in np.unique(ir[valid_bins]):
                vals = G[i] * G[mask][valid_bins][ir[valid_bins] == bin_idx]
                if len(vals) > 0:
                    covariance[bin_idx] += np.median(vals)
                    ncov[bin_idx] += 1
    covariance = np.where(ncov > 0, covariance / ncov, 0)
    covdist = np.arange(n_bins) * ds
    return covariance, covdist


def fit_exponential_covariance(X: np.ndarray, Y: np.ndarray, G: np.ndarray, 
                              covariance: np.ndarray, covdist: np.ndarray) -> Tuple[float, float]:
    '''
    Fit exponential covariance model parameters.
    
    Parameters
    ----------
    X           : X coordinates of observations
    Y           : Y coordinates of observations
    G           : Observation values
    covariance  : Empirical covariance values
    covdist     : Corresponding distances
        
    Returns
    -------
    C0          : Variance parameter
    D           : Correlation length parameter
    '''
    Dmax = np.sqrt((X.max() - X.min())**2 + (Y.max() - Y.min())**2)
    Step = np.sqrt((2 * np.pi * (Dmax / 2)**2) / len(X))
    C0 = np.var(G)  # Use variance instead of std^2
    s = covdist
    
    # Optimize over range parameter
    D_range = np.arange(Step, Dmax + Step, Step)
    errors = np.zeros_like(D_range)
    
    for i, D in enumerate(D_range):
        covp = C0 * np.exp(-s / D)
        errors[i] = np.sum((covp - covariance)**2)
    
    # Find D with minimum error
    best_idx = np.argmin(errors)
    Dbest = D_range[best_idx]
    
    return C0, Dbest


def fit_gaussian_covariance(X: np.ndarray, Y: np.ndarray, G: np.ndarray, 
                           covariance: np.ndarray, covdist: np.ndarray) -> Tuple[float, float]:
    '''
    Fit Gaussian covariance model parameters.
    
    Parameters
    ----------
    X           : X coordinates of observations
    Y           : Y coordinates of observations
    G           : Observation values
    covariance  : Empirical covariance values
    covdist     : Corresponding distances
        
    Returns
    -------
    C0          : Variance parameter
    D           : Correlation length parameter
    '''
    Dmax = np.sqrt((X.max() - X.min())**2 + (Y.max() - Y.min())**2)
    Step = np.sqrt((2 * np.pi * (Dmax / 2)**2) / len(X))
    C0 = np.var(G)  # Use variance instead of std^2
    s = covdist
    
    # Optimize over range parameter
    D_range = np.arange(Step, Dmax + Step, Step)
    errors = np.zeros_like(D_range)
    
    for i, D in enumerate(D_range):
        covp = C0 * np.exp(-(np.log(2) * s**2) / (D**2))
        errors[i] = np.sum((covp - covariance)**2)
    
    # Find D with minimum error
    best_idx = np.argmin(errors)
    Dbest = D_range[best_idx]
    
    return C0, Dbest


def lsc_exponential(Xi: np.ndarray, Yi: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                   C0: float, D: float, N: np.ndarray, G: np.ndarray) -> np.ndarray:
    '''
    Perform Least Squares Collocation with exponential covariance model.
    
    Parameters
    ----------
    Xi          : X coordinates of interpolation points
    Yi          : Y coordinates of interpolation points
    X           : X coordinates of observations
    Y           : coordinates of observations
    C0          : Variance parameter
    D           : Correlation length parameter
    N           : Noise variance for each observation
    G           : Observation values
        
    Returns
    -------
    SolG            : Interpolated values at interpolation points
    '''
    # Compute data-to-data distances (observation points)
    s2 = (X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2
    r = np.sqrt(s2)
    Czz = C0 * np.exp(-r / D)
    
    # Compute prediction-to-data distances (interpolation points to observation points)
    s2i = (Xi[:, None] - X[None, :])**2 + (Yi[:, None] - Y[None, :])**2
    ri = np.sqrt(s2i)
    Csz = C0 * np.exp(-ri / D)
    
    # Add noise to diagonal of covariance matrix
    Czz_noise = Czz + np.diag(N)
    
    # Solve LSC system
    # Using more stable approach with Cholesky decomposition if possible
    try:
        L = np.linalg.cholesky(Czz_noise)
        alpha = np.linalg.solve(L, G)
        beta = np.linalg.solve(L.T, alpha)
        SolG = Csz @ beta
    except np.linalg.LinAlgError:
        # Fall back to direct solve if Cholesky fails
        SolG = Csz @ np.linalg.solve(Czz_noise, G)
        
    return SolG


def lsc_gaussian(Xi: np.ndarray, Yi: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                C0: float, D: float, N: np.ndarray, G: np.ndarray) -> np.ndarray:
    '''
    Perform Least Squares Collocation with Gaussian covariance model.
    
    Parameters
    ----------
    Xi          : X coordinates of interpolation points
    Yi          : Y coordinates of interpolation points
    X           : X coordinates of observations
    Y           : Y coordinates of observations
    C0          : Variance parameter
    D           : Correlation length parameter
    N           : Noise variance for each observation
    G           : Observation values
        
    Returns
    -------
    SolG        : Interpolated values at interpolation points
    '''
    # Compute data-to-data distances
    s2 = (X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2
    r = np.sqrt(s2)
    Czz = C0 * np.exp(-(np.log(2) * r**2) / (D**2))
    
    # Compute prediction-to-data distances
    s2i = (Xi[:, None] - X[None, :])**2 + (Yi[:, None] - Y[None, :])**2
    ri = np.sqrt(s2i)
    Csz = C0 * np.exp(-(np.log(2) * ri**2) / (D**2))
    
    # Add noise to diagonal of covariance matrix
    Czz_noise = Czz + np.diag(N)
    
    # Solve LSC system
    # Using more stable approach with Cholesky decomposition if possible
    try:
        L = np.linalg.cholesky(Czz_noise)
        alpha = np.linalg.solve(L, G)
        beta = np.linalg.solve(L.T, alpha)
        SolG = Csz @ beta
    except np.linalg.LinAlgError:
        # Fall back to direct solve if Cholesky fails
        SolG = Csz @ np.linalg.solve(Czz_noise, G)
        
    return SolG