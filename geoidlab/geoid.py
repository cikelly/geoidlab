############################################################
# Utilities for geoid modelling                            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import xarray as xr

from geoidlab.utils.distances import haversine_fast, haversine_vectorized
from geoidlab.gravity import normal_gravity_somigliana
from geoidlab.constants import earth, resolve_ellipsoid
from geoidlab.stokes_func import Stokes4ResidualGeoid

from tqdm import tqdm

class ResidualGeoid:
    '''
    Initialize the ResidualGeoid class.
    '''
    VALID_METHODS = {'hg', 'wg', 'og', 'ml'}  # Valid integration methods
    VALID_WINDOW_MODES = {'fixed', 'cap', 'radius'}
    DEFAULT_WINDOW_MODE = 'cap'
    METHODS_DICT = {
        'hg': 'Heck & Gruninger',
        'wg': 'Wong & Gore',
        'og': 'Original Stokes\'',
        'ml': 'Meissl'
    }
    
    def __init__(
        self,
        res_anomaly: xr.Dataset,
        sph_cap: float = 1.0,
        sub_grid: tuple[float, float, float, float] = None,
        method: str = 'hg',
        ellipsoid: str = 'wgs84',
        nmax: int = None,
        window_mode: str = 'cap',
        fast: bool = False
    ) -> None:
        '''
        Initialize the ResidualGeoid class.
        
        Parameters
        ----------
        res_anomaly: gridded residual gravity anomalies
        sph_cap    : spherical cap for integration (degrees)
        sub_grid   : sub-grid to use for integration (min_lon, max_lon, min_lat, max_lat)
        method     : method for integration. Options are:
                        'hg' : Heck and Gruninger's modification
                        'wg' : Wong and Gore's modification
                        'og' : original Stokes' function
                        'ml' : Meissl's modification
        ellipsoid  : reference ellipsoid for normal gravity calculation
        nmax       : maximum degree of spherical harmonic expansion
        window_mode: window mode for integration. Options are: 'fixed' or 'cap'
        fast       : use the row-exact accelerated far-zone computation (cap/radius only)

        Returns
        -------
        None
        
        Reference
        ---------
        1. Hofmann-Wellenhof & Moritz (2005): Physical Geodesy (Section 2.21)
        
        Notes
        -----
        window_mode='cap' is the same as window_mode='radius'
        '''
        # Validate sub-grid
        if sub_grid is None:
            raise ValueError('sub_grid must be provided')
        if len(sub_grid) != 4:
            raise ValueError('sub_grid must be a tuple of 4 values: (min_lon, max_lon, min')
        if not all(isinstance(x, (int, float)) for x in sub_grid):
            raise ValueError('All sub_grid values must be numeric')
        if sub_grid[0] >= sub_grid[1] or sub_grid[2] >= sub_grid[3]:
            raise ValueError('sub_grid must be a tuple of 4 values: (min_lon, max_lon, min_lat, max_lat)')
        
        # Validate method
        if method.lower() not in self.VALID_METHODS:
            raise ValueError(f'Invalid method: {method}. Must be one of {sorted(self.VALID_METHODS)}')
        
        # Validate nmax 
        if method.lower() in {'hg', 'wg'}:
            if nmax is None:
                raise ValueError(f'nmax must be provided for {method} method')
            if not isinstance(nmax, int) or nmax <=0:
                raise ValueError(f'nmax must be a positive integer, got {nmax}')
        
        # Validate ellipsoid
        try:
            resolve_ellipsoid(ellipsoid)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid ellipsoid specification: {exc}') from exc
            
        # Validate res_anomaly
        if not isinstance(res_anomaly, xr.Dataset):
            raise TypeError('res_anomaly must be an xarray Dataset')
        if 'Dg' not in res_anomaly.data_vars:
            raise ValueError('res_anomaly must contain a \'Dg\' variable for gravity anomalies')
        if 'lon' not in res_anomaly.coords or 'lat' not in res_anomaly.coords:
            raise ValueError('res_anomaly must have \'lon\' and \'lat\' coordinates')
        
        # Validate window_mode
        if window_mode.lower() not in self.VALID_WINDOW_MODES:
            print(f'Warning: Invalid window_mode: \'{window_mode}\'. Setting window_mode=\'{self.DEFAULT_WINDOW_MODE}\'')
            window_mode = self.DEFAULT_WINDOW_MODE
            
        # Ensure sub_grid is within the bounds of res_anomaly
        min_lon, max_lon, min_lat, max_lat = sub_grid
        lon_min, lon_max = res_anomaly['lon'].min().item(), res_anomaly['lon'].max().item()
        lat_min, lat_max = res_anomaly['lat'].min().item(), res_anomaly['lat'].max().item()
        
        if (min_lon < lon_min or max_lon > lon_max or min_lat < lat_min or max_lat > lat_max):
            raise ValueError(f'sub_grid {sub_grid} is outside res_anomaly bounds (lon: [{lon_min}, {lon_max}], lat: [{lat_min}, {lat_max}])')
        
        # Validate spherical cap (sph_cap)
        if not isinstance(sph_cap, (int, float)) or sph_cap <= 0:
            raise ValueError(f'sph_cap must be a positive number, got {sph_cap}')
        
        # Store validated parameters
        self.sub_grid = sub_grid
        self.res_anomaly = res_anomaly
        self.sph_cap = sph_cap
        self.method = method
        self.ellipsoid = ellipsoid
        self.nmax = nmax
        self.window_mode = window_mode
        self.fast = fast

        # Grid information
        self.Lon, self.Lat = np.meshgrid(res_anomaly['lon'].values, res_anomaly['lat'].values)
        self.nrows, self.ncols = res_anomaly['Dg'].shape
        self.dphi = (lat_max - lat_min) / (self.nrows - 1)
        self.dlam = (lon_max - lon_min) / (self.ncols - 1)
        
        # Extract sub-grid residual anomalies
        self.res_anomaly_P = self.res_anomaly.sel(
            lon=slice(min_lon, max_lon), 
            lat=slice(min_lat, max_lat)
        )
        
        self.LonP, self.LatP = np.meshgrid(self.res_anomaly_P['lon'].values, self.res_anomaly_P['lat'].values)
        self.nrows_P, self.ncols_P = self.res_anomaly_P['Dg'].shape
        self.phip = np.radians(self.LatP)
        self.lonp = np.radians(self.LonP)
        
        # Normal gravity (Somigliana's method)
        self.gamma_0  = normal_gravity_somigliana(phi=self.LatP, ellipsoid=self.ellipsoid)
        self.gamma_0 *= 1e5  # Convert to mGal
        
        # Precompute constants
        self.R = earth()['radius'] # Earth radius
        
    def stokes_kernel(self) -> np.ndarray:
        '''
        Compute Stokes' kernel based on the selected method
        
        Returns
        -------
        S_k     : Stokes' kernel values
        '''
        method_map = {
            'og': lambda: self.stokes_calculator.stokes()[0],
            'wg': lambda: self.stokes_calculator.wong_and_gore(),
            'hg': lambda: self.stokes_calculator.heck_and_gruninger(),
            'ml': lambda: self.stokes_calculator.meissl()
        }
        
        S_k = method_map[self.method]()
        
        # Set NaN and +/- inf values to 0
        S_k = np.nan_to_num(S_k, nan=0.0, posinf=0.0, neginf=0.0)
        
        return S_k

    def _compute_far_zone_point(
        self,
        i: int,
        j: int,
        row_p: int,
        col_p: int,
        psi0: float,
        dphi_2: float,
        dlam_2: float
    ) -> float:
        '''Single-point far-zone contribution using the original cap method.'''
        mask, (row_min, row_max, col_min, col_max) = self.get_cap_window(row_p, col_p)
        win_Dg  = self.res_anomaly['Dg'].values[row_min:row_max, col_min:col_max]
        win_phi = np.radians(self.Lat[row_min:row_max, col_min:col_max])
        win_lon = np.radians(self.Lon[row_min:row_max, col_min:col_max])
        win_Dg  = np.where(mask, win_Dg,  np.nan)
        win_phi = np.where(mask, win_phi, np.nan)
        win_lon = np.where(mask, win_lon, np.nan)
        lat1 = win_phi - dphi_2
        lat2 = win_phi + dphi_2
        lon1 = win_lon - dlam_2
        lon2 = win_lon + dlam_2
        A_k = self.R**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))
        A_k = np.where(np.isfinite(win_Dg), A_k, np.nan)
        self.stokes_calculator = Stokes4ResidualGeoid(
            lonp=self.lonp[i, j],
            latp=self.phip[i, j],
            lon=win_lon,
            lat=win_phi,
            psi0=psi0,
            nmax=self.nmax
        )
        S_k = self.stokes_kernel()
        S_k = np.where(mask, S_k, np.nan)
        c_k = A_k * S_k
        return np.nansum(c_k * win_Dg) / (4 * np.pi * self.gamma_0[i, j] * self.R)

    def _compute_far_zone_row_exact(
        self,
        psi0: float,
        dphi_2: float,
        dlam_2: float
    ) -> np.ndarray:
        '''
        Accelerated row-exact far-zone contribution.

        For a regular geographic grid every computation point in row i shares the
        same latitude, so the Stokes weight matrix W_i(a, b) is identical for all
        columns j in that row.  The matrix is built once per row and the gravity-
        anomaly window is shifted column-by-column without recomputing the kernel.

        Rows and columns whose cap window is clipped by the grid boundary fall back
        to _compute_far_zone_point so results are identical to the original method.

        '''
        N_far = np.zeros_like(self.N_inner)

        lat     = self.res_anomaly['lat'].values
        lon     = self.res_anomaly['lon'].values
        lat_sub = self.res_anomaly_P['lat'].values
        lon_sub = self.res_anomaly_P['lon'].values
        row_start = np.where(lat == lat_sub[0])[0][0]
        col_start = np.where(lon == lon_sub[0])[0][0]

        dn = int(np.ceil(self.sph_cap / self.dphi))

        for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
            row_p = row_start + i
            lat_i = self.Lat[row_p, col_start]
            phi_i = np.radians(lat_i)
            dm    = int(np.ceil(self.sph_cap / (self.dlam * np.cos(np.radians(lat_i)))))

            # Rows where the cap window is clipped at the top or bottom grid edge
            if row_p - dn < 0 or row_p + dn + 1 > self.nrows:
                for j in range(self.ncols_P):
                    N_far[i, j] = self._compute_far_zone_point(
                        i, j, row_p, col_start + j, psi0, dphi_2, dlam_2
                    )
                continue

            # --- Build W_i(a, b) once for this row (Eq. 11 of PDF) ---
            a_vals = np.arange(-dn, dn + 1)         # row offsets, length 2*dn+1
            b_vals = np.arange(-dm, dm + 1)         # col offsets, length 2*dm+1
            A_grid, B_grid = np.meshgrid(a_vals, b_vals, indexing='ij')  # (2*dn+1, 2*dm+1)

            row_idx = row_p + A_grid

            # Integration-cell latitudes in degrees and radians
            lat_k_deg = self.Lat[row_idx, col_start]    # (2*dn+1, 2*dm+1)
            phi_k     = np.radians(lat_k_deg)

            # Relative longitude offsets (Eq. 13: cos ψ depends only on b·Δλ)
            b_lam_deg = B_grid * self.dlam
            b_lam_rad = np.radians(b_lam_deg)

            # Cell areas — depend only on integration-cell latitude (Eq. 4 of PDF)
            A_k = self.R**2 * 2 * dlam_2 * np.abs(
                np.sin(phi_k + dphi_2) - np.sin(phi_k - dphi_2)
            )

            # Stokes kernel with lonp=0 reference: algebraically equivalent to the
            # original because the kernel depends only on the spherical distance ψ,
            # and cos ψ_i(a,b) = sin φ_i sin φ_{ki+a} + cos φ_i cos φ_{ki+a} cos(b Δλ)
            # (Eq. 13) is identical to the general formula with the longitude origin
            # shifted to the computation point.
            self.stokes_calculator = Stokes4ResidualGeoid(
                lonp=0.0,
                latp=phi_i,
                lon=b_lam_rad,
                lat=phi_k,
                psi0=psi0,
                nmax=self.nmax
            )
            S_k = self.stokes_kernel()

            # Cap mask — same haversine logic as get_cap_window
            distances = haversine_vectorized(0.0, lat_i, b_lam_deg, lat_k_deg, 'deg', 'deg')
            chi = distances <= self.sph_cap

            W = np.where(chi, A_k * S_k, np.nan)   # (2*dn+1, 2*dm+1)

            # Shift W across all columns in this row
            for j in range(self.ncols_P):
                col_p = col_start + j

                # Columns where the cap window is clipped at the left or right grid edge
                if col_p - dm < 0 or col_p + dm + 1 > self.ncols:
                    N_far[i, j] = self._compute_far_zone_point(
                        i, j, row_p, col_p, psi0, dphi_2, dlam_2
                    )
                    continue

                win_Dg = self.res_anomaly['Dg'].values[
                    row_p - dn : row_p + dn + 1,
                    col_p - dm : col_p + dm + 1
                ]
                N_far[i, j] = np.nansum(W * win_Dg) / (4 * np.pi * self.gamma_0[i, j] * self.R)

        return N_far

    def compute_geoid(self) -> np.ndarray:
        '''
        Compute the residual geoid height
        
        Returns
        -------
        N_res  : Residual geoid height in meters
        '''
        cosphip = np.cos(self.phip)

        # Near zone computation
        print('Computing inner zone...')
        self.N_inner = self.R / self.gamma_0 * np.sqrt(
            cosphip * np.radians(self.dphi) * np.radians(self.dlam) / np.pi
        ) * self.res_anomaly_P['Dg'].values
        print('Inner zone computation completed.')
        
        # Far zone computation
        print(f'Computing far zone using: {self.METHODS_DICT[self.method]} method...')
        self.N_far = np.zeros_like(self.N_inner)
        psi0 = np.radians(self.sph_cap)
        
        # Precompute np.radians(self.dphi) / 2 and np.radians(self.dlam) / 2
        dphi_2 = np.radians(self.dphi) / 2
        dlam_2 = np.radians(self.dlam) / 2
        
        if self.window_mode == 'fixed':
            # Sub-window for integration
            dn = round(self.nrows - self.nrows_P) + 1
            dm = round(self.ncols - self.ncols_P) + 1
            n1 = 0
            n2 = dn
            
            for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
                m1 = 0
                m2 = dm
                for j in range(self.ncols_P):
                    # Extract window data
                    win_Dg = self.res_anomaly['Dg'].values[n1:n2, m1:m2]
                    win_phi = np.radians(self.Lat[n1:n2, m1:m2])
                    win_lon = np.radians(self.Lon[n1:n2, m1:m2])
                    
                    # Compute surface area on the sphere
                    lat1 = win_phi - dphi_2
                    lat2 = win_phi + dphi_2
                    lon1 = win_lon - dlam_2
                    lon2 = win_lon + dlam_2
                    A_k = self.R**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))
                    
                    self.stokes_calculator = Stokes4ResidualGeoid(
                        lonp=self.lonp[i, j],
                        latp=self.phip[i, j],
                        lon=win_lon,
                        lat=win_phi,
                        psi0=psi0,
                        nmax=self.nmax
                    )
                    
                    # S_k = self.stokes_calculator.stokes()[0]
                    S_k = self.stokes_kernel()
                    
                    # Spherical Distance
                    sd = haversine_fast(
                        lon1=self.lonp[i, j],
                        lat1=self.phip[i, j],
                        lon2=win_lon,
                        lat2=win_phi,
                        in_unit='rad',
                        out_unit='deg'
                    )
                    
                    
                    sd[sd > self.sph_cap] = np.nan
                    
                    # Mask S_k for points beyond self.scap
                    S_k[np.isnan(sd)] = np.nan
                    
                    # Outer (far) zone
                    c_k = A_k * S_k
                    self.N_far[i, j] = np.nansum(c_k * win_Dg) * 1 / (4 * np.pi * self.gamma_0[i, j] * self.R)
                    
                    # Move window
                    m1 += 1
                    m2 += 1
                # Move window
                n1 += 1
                n2 += 1
        else:
            if self.fast:
                self.N_far = self._compute_far_zone_row_exact(psi0, dphi_2, dlam_2)
            else:
                # Find sub-grid offset in full grid
                lat = self.res_anomaly['lat'].values
                lon = self.res_anomaly['lon'].values
                lat_sub = self.res_anomaly_P['lat'].values
                lon_sub = self.res_anomaly_P['lon'].values
                row_start = np.where(lat == lat_sub[0])[0][0]
                col_start = np.where(lon == lon_sub[0])[0][0]

                for i in tqdm(range(self.nrows_P), desc='Computing far zone contribution'):
                    for j in range(self.ncols_P):
                        # Map sub-grid to full grid indices
                        row_p = row_start + i
                        col_p = col_start + j

                        # Get cap window bounds and mask
                        mask, (row_min, row_max, col_min, col_max) = self.get_cap_window(row_p, col_p)

                        # Extract window data
                        win_Dg = self.res_anomaly['Dg'].values[row_min:row_max, col_min:col_max]
                        win_phi = np.radians(self.Lat[row_min:row_max, col_min:col_max])
                        win_lon = np.radians(self.Lon[row_min:row_max, col_min:col_max])

                        # Apply mask to windowed data
                        win_Dg = np.where(mask, win_Dg, np.nan)
                        win_phi = np.where(mask, win_phi, np.nan)
                        win_lon = np.where(mask, win_lon, np.nan)

                        # Compute surface area on the sphere
                        lat1 = win_phi - dphi_2
                        lat2 = win_phi + dphi_2
                        lon1 = win_lon - dlam_2
                        lon2 = win_lon + dlam_2
                        A_k = self.R**2 * np.abs(lon2 - lon1) * np.abs(np.sin(lat2) - np.sin(lat1))
                        A_k = np.where(np.isfinite(win_Dg), A_k, np.nan)

                        self.stokes_calculator = Stokes4ResidualGeoid(
                            lonp=self.lonp[i, j],
                            latp=self.phip[i, j],
                            lon=win_lon,
                            lat=win_phi,
                            psi0=psi0,
                            nmax=self.nmax
                        )

                        S_k = self.stokes_kernel()
                        # Apply mask to S_k (use mask from get_cap_window)
                        S_k = np.where(mask, S_k, np.nan)

                        # Outer (far) zone
                        c_k = A_k * S_k
                        self.N_far[i, j] = np.nansum(c_k * win_Dg) * 1 / (4 * np.pi * self.gamma_0[i, j] * self.R)

        print('Far zone computation completed.')
        N_res = self.N_inner + self.N_far
        self.N_res = N_res

        return N_res

    def get_cap_window(self, row_p, col_p) -> tuple[bool, tuple[int, int, int, int]]:
        '''
        Get indices of points within spherical cap

        Parameters
        ----------
        row_p       : row index of computation point
        col_p       : column index of computation point
        
        Returns
        -------
        mask        : boolean mask of points within spherical cap
        bounds      : (row_min, row_max, col_min, col_max) of window
        
        Notes
        -----
        This approach is a hybrid of a rectangular window and a circular window.
        We start with a rectangular window adjusted for the latitude of the computation point.
        Then we refined the selection by excluding points outside the spherical cap.
        '''
        # Latitude of computation point
        lat_p = self.Lat[row_p, col_p]
        lon_p = self.Lon[row_p, col_p]
        
        # Estimate window size based on latitude
        dn = int(np.ceil(self.sph_cap / self.dphi))
        dm = int(np.ceil(self.sph_cap / (self.dlam * np.cos(np.radians(lat_p)))))
        
        # Define window bounds
        row_min = max(0, row_p - dn)
        row_max = min(self.nrows, row_p + dn + 1)
        col_min = max(0, col_p - dm)
        col_max = min(self.ncols, col_p + dm + 1)
        
        # Extract window coordinates
        lat_window = self.Lat[row_min:row_max, col_min:col_max]
        lon_window = self.Lon[row_min:row_max, col_min:col_max] 
        
        # Compute distances within window
        distances = haversine_vectorized(lon_p, lat_p, lon_window, lat_window, 'deg', 'deg')
        
        # Mask points within spherical cap
        mask = distances <= self.sph_cap
        
        return mask, (row_min, row_max, col_min, col_max)
