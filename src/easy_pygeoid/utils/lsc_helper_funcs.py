import numpy as np
from typing import Union, Tuple, List, Optional

def compute_spatial_covariance(
    X: np.ndarray, 
    Y: np.ndarray, 
    G: np.ndarray,
    max_points: Optional[int] = 10000,
    chunk_size: Optional[int] = 1000,
    use_chunking: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute empirical covariance of 2D spatial data with optional chunking.
    
    Parameters
    ----------
    X           : X coordinates of observations
    Y           : Y coordinates of observations
    G           : Observation values
    max_points  : Maximum number of points to process at once (default: 10000)
    chunk_size  : Size of chunks for processing large datasets (default: 1000)
    use_chunking: Whether to process data in chunks (default: True)
    
    Returns
    -------
    covariance : Empirical covariance values
    covdist    : Corresponding distances
    '''
    # Subsample if dataset is too large
    if max_points and len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X[idx]
        Y = Y[idx]
        G = G[idx]
    
    smax = np.sqrt((X.max() - X.min())**2 + (Y.max() - Y.min())**2)
    ds = np.sqrt((2 * np.pi * (smax / 2)**2) / len(X))
    n_bins = int(np.round(smax / ds)) + 2
    covariance = np.zeros(n_bins)
    ncov = np.zeros(n_bins)
    
    if use_chunking:
        # Process data in chunks
        n_chunks = int(np.ceil(len(G) / chunk_size))
        for chunk in range(n_chunks):
            start_idx = chunk * chunk_size
            end_idx = min((chunk + 1) * chunk_size, len(G))
            
            # Process each point in the chunk
            for i in range(start_idx, end_idx):
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
    else:
        # Process all data at once
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        r = np.sqrt(dx**2 + dy**2)
        
        # Skip self-distance
        mask = (r > 0) & (r < smax)
        ir = np.round(r[mask] / ds).astype(int)
        
        # Only process bins that are within range
        valid_bins = ir < n_bins
        if np.any(valid_bins):
            values_product = G[:, None] * G[None, :]
            values_product = values_product[mask][valid_bins]
            np.add.at(covariance, ir[valid_bins], values_product)
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
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-12
    covariance = np.where(ncov > 0, covariance / (ncov + epsilon), 0)
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


def lsc_exponential(
    Xi: np.ndarray, 
    Yi: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    C0: float, 
    D: float, 
    N: np.ndarray, 
    G: np.ndarray,
    n_jobs: int = -1,
    chunk_size: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
    use_chunking: bool = True
) -> np.ndarray:
    '''
    Perform Least Squares Collocation with exponential covariance model.
    Includes parallel processing and caching optimizations.
    
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
    n_jobs      : Number of parallel jobs (-1 for all cores)
    chunk_size  : Size of chunks for parallel processing (default: 1000)
    cache_dir   : Directory to store cached computations (None for no caching)
    use_chunking: Whether to process data in chunks (default: True)
        
    Returns
    -------
    SolG        : Interpolated values at interpolation points
    '''
    from joblib import Parallel, delayed, Memory
    import os
    
    # Setup caching if cache_dir is provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        memory = Memory(cache_dir, verbose=0)
        cached_exp = memory.cache(np.exp)
    else:
        cached_exp = np.exp

    if use_chunking:
        # Process in chunks with parallel computation
        def compute_chunk_covariance(start_idx, end_idx):
            chunk_X = X[start_idx:end_idx]
            chunk_Y = Y[start_idx:end_idx]
            s2 = (chunk_X[:, None] - X[None, :])**2 + (chunk_Y[:, None] - Y[None, :])**2
            r = np.sqrt(s2)
            return C0 * cached_exp(-r / D)
        
        # Process observation points in parallel chunks
        n_obs = len(X)
        chunk_indices = [(i, min(i + chunk_size, n_obs)) 
                        for i in range(0, n_obs, chunk_size)]
        
        Czz_chunks = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk_covariance)(start_idx, end_idx)
            for start_idx, end_idx in chunk_indices
        )
        Czz = np.vstack(Czz_chunks)
        
        # Compute prediction-to-data distances in parallel chunks
        def compute_interp_chunk(start_idx, end_idx):
            chunk_Xi = Xi[start_idx:end_idx]
            chunk_Yi = Yi[start_idx:end_idx]
            s2i = (chunk_Xi[:, None] - X[None, :])**2 + (chunk_Yi[:, None] - Y[None, :])**2
            ri = np.sqrt(s2i)
            return C0 * cached_exp(-ri / D)
        
        # Process interpolation points in parallel chunks
        n_interp = len(Xi)
        interp_indices = [(i, min(i + chunk_size, n_interp)) 
                         for i in range(0, n_interp, chunk_size)]
        
        Csz_chunks = Parallel(n_jobs=n_jobs)(
            delayed(compute_interp_chunk)(start_idx, end_idx)
            for start_idx, end_idx in interp_indices
        )
        Csz = np.vstack(Csz_chunks)
    else:
        # Process all data at once
        s2 = (X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2
        r = np.sqrt(s2)
        Czz = C0 * cached_exp(-r / D)
        
        s2i = (Xi[:, None] - X[None, :])**2 + (Yi[:, None] - Y[None, :])**2
        ri = np.sqrt(s2i)
        Csz = C0 * cached_exp(-ri / D)
    
    # Add noise to diagonal of covariance matrix
    Czz_noise = Czz + np.diag(N)
    
    # Solve LSC system with optimized linear algebra
    try:
        # Try Cholesky decomposition first (more stable)
        L = np.linalg.cholesky(Czz_noise)
        alpha = np.linalg.solve(L, G)
        beta = np.linalg.solve(L.T, alpha)
        SolG = Csz @ beta
    except np.linalg.LinAlgError:
        # Fall back to direct solve if Cholesky fails
        SolG = Csz @ np.linalg.solve(Czz_noise, G)
    
    # Clear cached computations if using cache
    if cache_dir:
        memory.clear()
    
    return SolG


def lsc_gaussian(
    Xi: np.ndarray, 
    Yi: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    C0: float, 
    D: float, 
    N: np.ndarray, 
    G: np.ndarray,
    n_jobs: int = -1,
    chunk_size: Optional[int] = 1000,
    cache_dir: Optional[str] = None,
    use_chunking: bool = True
) -> np.ndarray:
    '''
    Perform Least Squares Collocation with Gaussian covariance model.
    Includes parallel processing and caching optimizations.
    
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
    n_jobs      : Number of parallel jobs (-1 for all cores)
    chunk_size  : Size of chunks for parallel processing (default: 1000)
    cache_dir   : Directory to store cached computations (None for no caching)
    use_chunking: Whether to process data in chunks (default: True)
        
    Returns
    -------
    SolG        : Interpolated values at interpolation points
    '''
    from joblib import Parallel, delayed, Memory
    import os
    
    # Setup caching if cache_dir is provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        memory = Memory(cache_dir, verbose=0)
        cached_exp = memory.cache(np.exp)
    else:
        cached_exp = np.exp

    if use_chunking:
        # Process in chunks with parallel computation
        def compute_chunk_covariance(start_idx, end_idx):
            chunk_X = X[start_idx:end_idx]
            chunk_Y = Y[start_idx:end_idx]
            s2 = (chunk_X[:, None] - X[None, :])**2 + (chunk_Y[:, None] - Y[None, :])**2
            return C0 * cached_exp(-(np.log(2) * s2) / (D**2))
        
        # Process observation points in parallel chunks
        n_obs = len(X)
        chunk_indices = [(i, min(i + chunk_size, n_obs)) 
                        for i in range(0, n_obs, chunk_size)]
        
        Czz_chunks = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk_covariance)(start_idx, end_idx)
            for start_idx, end_idx in chunk_indices
        )
        Czz = np.vstack(Czz_chunks)
        
        # Compute prediction-to-data distances in parallel chunks
        def compute_interp_chunk(start_idx, end_idx):
            chunk_Xi = Xi[start_idx:end_idx]
            chunk_Yi = Yi[start_idx:end_idx]
            s2i = (chunk_Xi[:, None] - X[None, :])**2 + (chunk_Yi[:, None] - Y[None, :])**2
            return C0 * cached_exp(-(np.log(2) * s2i) / (D**2))
        
        # Process interpolation points in parallel chunks
        n_interp = len(Xi)
        interp_indices = [(i, min(i + chunk_size, n_interp)) 
                         for i in range(0, n_interp, chunk_size)]
        
        Csz_chunks = Parallel(n_jobs=n_jobs)(
            delayed(compute_interp_chunk)(start_idx, end_idx)
            for start_idx, end_idx in interp_indices
        )
        Csz = np.vstack(Csz_chunks)
    else:
        # Process all data at once
        s2 = (X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2
        Czz = C0 * cached_exp(-(np.log(2) * s2) / (D**2))
        
        s2i = (Xi[:, None] - X[None, :])**2 + (Yi[:, None] - Y[None, :])**2
        Csz = C0 * cached_exp(-(np.log(2) * s2i) / (D**2))
    
    # Add noise to diagonal of covariance matrix
    Czz_noise = Czz + np.diag(N)
    
    # Solve LSC system with optimized linear algebra
    try:
        # Try Cholesky decomposition first (more stable)
        L = np.linalg.cholesky(Czz_noise)
        alpha = np.linalg.solve(L, G)
        beta = np.linalg.solve(L.T, alpha)
        SolG = Csz @ beta
    except np.linalg.LinAlgError:
        # Fall back to direct solve if Cholesky fails
        SolG = Csz @ np.linalg.solve(Czz_noise, G)
    
    # Clear cached computations if using cache
    if cache_dir:
        memory.clear()
    
    return SolG