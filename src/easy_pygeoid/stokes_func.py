############################################################
# Stokes' function and its modifications                   #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import numpy as np

from .legendre import legendre_poly

class Stokes:
    '''
    Class for Stokes' function and its modifications
    '''
    def __init__(
        self,
        comp_point=None, 
        int_points=None, 
        psi=None,
        psi0=None, 
        nmax=None
    ) -> None:
        '''
        Initialize the Stokes class.
        
        Parameters
        ----------
        comp_point : array-like, shape (2,)
                       [lon, lat] of computation point in degrees
        int_points : array-like, shape (n, 2)
                       [lon, lat] of integration points in degrees
        psi        : array-like, spherical distances in radians
        psi0       : float, spherical distance of the spherical cap in radians
        nmax       : int, maximum degree of expansion
        
        Returns
        -------
        None
        '''
        self.comp_point = comp_point
        self.int_points = int_points
        self.psi  = psi
        self.psi0 = psi0
        self.nmax = nmax
    
    def _validate_comp_int_points(self) -> bool:
        '''Validate computation and integration points are provided'''
        if self.comp_point is None or self.int_points is None:
            if self.psi is not None:
                return False  # Use psi variant instead
            raise ValueError('comp_point and int_points must be provided')
        return True
        
    def stokes(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Implements the original Stokes' kernel
        
        Returns
        -------
        S         : Stokes' function
        cos_psi   : Cosine of spherical distance
        '''
        # Check if we should use direct spherical distance calculation instead
        if not self._validate_comp_int_points():
            return self.stokes_psi()
            
        lonp, latp = np.array(self.comp_point)
        lon, lat = np.array(self.int_points)[:, 0], np.array(self.int_points)[:, 1]
        lon, lat, lonp, latp = np.radians(lon), np.radians(lat), np.radians(lonp), np.radians(latp)
        
        # Calculate cos_dlam using spherical trigonometry
        cos_dlam = np.cos(lon) * np.cos(lonp) + np.sin(lon) * np.sin(lonp)
        
        # Calculate cos_psi using the spherical law of cosines
        cos_psi = np.sin(latp) * np.sin(lat) + np.cos(latp) * np.cos(lat) * cos_dlam  
        
        # Calculate sin^2(psi/2) using the Haversine formula
        sin2_psi_2 = np.sin((latp - lat)/2)**2 + np.cos(latp) * np.cos(lat) * np.sin((lonp - lon)/2)**2 
        
        # Calculate Stokes' function S
        S = self._compute_stokes(sin2_psi_2, cos_psi)
        
        return S, cos_psi

    def _compute_stokes(self, sin2_psi_2, cos_psi) -> np.ndarray[float]:
        '''Core computation of Stokes' function given sin²(ψ/2) and cos(ψ)'''
        log_arg = np.sqrt(sin2_psi_2) + sin2_psi_2
        return np.where(
            log_arg <= 0,
            0.0,
            1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - 3 * cos_psi * np.log(log_arg)
        )
        
    def stokes_psi(self, custom_psi=None, use_custom=False) -> tuple[np.ndarray, np.ndarray]:
        '''
        Original Stokes' function when spherical distance is given
        
        Parameters
        ----------
        custom_psi : specify a custom spherical distance in radians
        use_custom : if True, use custom_psi instead of self.psi
        
        Returns
        -------
        S         : Stokes' function
        cos_psi   : Cosine of spherical distance
        '''
        psi_to_use = custom_psi if use_custom else self.psi
        
        if psi_to_use is None:
            raise ValueError('psi must be provided')
        
        cos_psi = np.cos(psi_to_use)
        
        # Calculate sin^2(psi/2) using the relationship sin^2(psi/2) = (1 - cos(psi)) / 2
        sin2_psi_2 = (1 - cos_psi) / 2
        
        # Calculate Stokes' function S
        S = self._compute_stokes(sin2_psi_2, cos_psi)
            
        return S, cos_psi
    
    def _validate_meissl(self) -> bool:
        '''Validate parameters for Meissl's modification'''
        if not self._validate_comp_int_points():
            if self.psi is not None and self.psi0 is not None:
                return False  # Use psi variant instead
            if self.psi is not None:
                raise ValueError('psi0 must be provided for Meissl modification')
        elif self.psi0 is None:
            raise ValueError('psi0 must be provided for Meissl modification')
        return True
    
    def meissl(self) -> np.ndarray[float]:
        '''
        Meissl's modification of Stokes' function
        
        Returns
        -------
        S_m       : Modified Stokes' function
        '''
        if not self._validate_meissl():
            return self.meissl_psi()
        
        S, _ = self.stokes()
        
        # Calculate Stokes' function at cap boundary
        temp_stokes = Stokes(comp_point=[0, np.degrees(self.psi0)], int_points=np.array([[0, 0]]))
        S_0, _ = temp_stokes.stokes()

        return S - S_0
    
    def meissl_psi(self, custom_psi=None, use_custom=False) -> np.ndarray[float]:
        '''
        Meissl's modification of Stokes function when spherical distance is given
        
        Parameters
        ----------
        custom_psi : specify a custom spherical distance in radians
        use_custom : if True, use custom_psi instead of self.psi
        
        Returns
        -------
        S_m       : Modified Stokes' function
        '''
        S, _ = self.stokes_psi()
        
        psi0_to_use = custom_psi if use_custom else self.psi0
        if psi0_to_use is None:
            raise ValueError('psi0 must be provided')
        
        # Calculate Stokes' function at cap boundary
        temp_stokes = Stokes(psi=np.array([psi0_to_use]))
        S_w0, _ = temp_stokes.stokes_psi()
        
        return S - S_w0[0]
    
    def _validate_wong_gore(self) -> bool:
        '''Validate parameters for Wong and Gore's modification'''
        if not self._validate_comp_int_points():
            if self.psi is not None and self.nmax is not None:
                return False  # Use psi variant instead
            if self.psi is not None:
                raise ValueError('nmax must be provided for Wong and Gore modification')
        elif self.nmax is None:
            raise ValueError('nmax must be provided for Wong and Gore modification')
        return True
    
    def wong_and_gore(self) -> np.ndarray[float]:
        '''
        Wong and Gore's modification of Stokes' function
        
        Returns
        -------
        S_wg      : Wong and Gore's modification of Stokes' function
        '''
        if not self._validate_wong_gore():
            return self.wong_and_gore_psi()
        
        S, cos_psi = self.stokes()
        
        # Wong and Gore's modification
        return self._compute_wong_gore(S, cos_psi)

    def _compute_wong_gore(self, S, cos_psi) -> np.ndarray[float]:
        '''Core computation for Wong and Gore's modification'''
        # Flatten cos_psi to handle multi-dimensional arrays
        cos_psi_flat = cos_psi.ravel()
        S_flat = S.ravel()
        S_wg_flat = np.zeros_like(cos_psi_flat)

        for i, t in enumerate(cos_psi_flat):
            # Ensure t is within valid range for Legendre polynomials
            t_clipped = np.clip(t, -1.0, 1.0)
            Pn = legendre_poly(t=t_clipped, nmax=self.nmax)
            sum_term = sum((2 * n + 1) / (n - 1) * Pn[n] for n in range(2, self.nmax + 1))
            S_wg_flat[i] = S_flat[i] - sum_term

        # Reshape back to original dimensions
        return S_wg_flat.reshape(cos_psi.shape)
        
    def wong_and_gore_psi(self, custom_psi=None, use_custom=False) -> np.ndarray[float]:
        '''
        Wong and Gore's modification of Stokes' function when spherical distance is given
        
        Parameters
        ----------
        custom_psi : specify a custom spherical distance in radians
        use_custom : if True, use custom_psi instead of self.psi
        
        Returns
        -------
        S_wg      : Modified Stokes' function
        '''
        if self.nmax is None:
            raise ValueError('nmax must be provided')
            
        S, cos_psi = self.stokes_psi(custom_psi=custom_psi, use_custom=use_custom)
        
        # Wong and Gore's modification
        return self._compute_wong_gore(S, cos_psi)
    
    def _validate_heck_gruninger(self) -> bool:
        '''Validate parameters for Heck and Gruninger's modification'''
        if not self._validate_comp_int_points():
            if self.psi is not None and self.psi0 is not None and self.nmax is not None:
                return False  # Use psi variant instead
            missing = []
            if self.psi is None:
                missing.append('psi')
            if self.psi0 is None:
                missing.append('psi0')
            if self.nmax is None:
                missing.append('nmax')
            if missing:
                raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        elif self.psi0 is None or self.nmax is None:
            missing = []
            if self.psi0 is None:
                missing.append('psi0')
            if self.nmax is None:
                missing.append('nmax')
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        return True
    
    def heck_and_gruninger(self) -> np.ndarray[float]:
        '''
        Heck and Gruninger's modification of Stokes' function
        
        Returns
        -------
        S_hg      : Heck and Gruninger's modification of Stokes' function
        '''
        if not self._validate_heck_gruninger():
            return self.heck_and_gruninger_psi()
        
        # Wong and Gore modification
        S_wg = self.wong_and_gore()
        
        # Calculate at cap boundary
        # Stokes' function for a spherical cap (psi_0)
        temp_stokes = Stokes(comp_point=[0, np.degrees(self.psi0)], int_points=np.array([[0, 0]]))
        S_0, cos_psi_0 = temp_stokes.stokes()
        
        # Wong and Gore for spherical cap (psi_0)
        Pn = legendre_poly(t=cos_psi_0, nmax=self.nmax)
        S_wgL = 0
        for n in range(2, self.nmax + 1):
            S_wgL += (2 * n + 1) / (n - 1) * Pn[n]
        
        # Heck and Gruninger
        S_hg = S_wg - (S_0 - S_wgL)
        
        return S_hg
    
    def heck_and_gruninger_psi(self, custom_psi=None, use_custom=False) -> np.ndarray[float]:
        '''
        Heck and Gruninger's modification of Stokes' function when spherical distance is given
        
        Parameters
        ----------
        custom_psi : specify a custom spherical distance in radians
        use_custom : if True, use custom_psi instead of self.psi
        
        Returns
        -------
        S_hg      : Heck and Gruninger's modification of Stokes' function
        '''
        if self.psi is None or self.psi0 is None or self.nmax is None:
            missing = []
            if self.psi is None:
                missing.append('psi')
            if self.psi0 is None:
                missing.append('psi0')
            if self.nmax is None:
                missing.append('nmax')
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        S_wg = self.wong_and_gore_psi()
        
        psi0_to_use = custom_psi if use_custom else self.psi0
        
        # Calculate at cap boundary
        temp_stokes = Stokes(psi=np.array([psi0_to_use]), nmax=self.nmax)
        S_0, _ = temp_stokes.stokes_psi()
        S_wg0 = temp_stokes.wong_and_gore_psi()
        
        # Heck and Gruninger's modification
        S_hg = S_wg - (S_0[0] - S_wg0[0])
        
        return S_hg
        
def stokes(comp_point, int_points) -> tuple:
    '''
    Calculate the Original Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    
    Returns
    -------
    S         : Stokes' function
    cos_psi   : Cosine of spherical distance
    
    Notes
    -----
    1. https://en.wikipedia.org/wiki/Haversine_formula
    2. cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
    3. Spherical cosines
    4. Estimate sin2_psi_2 = sin^2(psi/2) using Haversine formula (Note 1)
    5. Physical Geodesy (2nd Edition) Page 104, Equation 2–305
    '''
    lonp, latp = np.array(comp_point)
    lon, lat = np.array(int_points[:, 0]), np.array(int_points[:, 1])
    
    lon, lat, lonp, latp = np.radians(lon), np.radians(lat), np.radians(lonp), np.radians(latp)
    
    # Calculate cos_dlam using spherical trigonometry (Note 2)
    cos_dlam = np.cos(lon) * np.cos(lonp) + np.sin(lon) * np.sin(lonp)  
    
    # Calculate cos_psi using the spherical law of cosines (Note 3)
    cos_psi = np.sin(latp) * np.sin(lat) + np.cos(latp) * np.cos(lat) * cos_dlam  
    
    # Calculate sin^2(psi/2) using the Haversine formula (Note 4)
    sin2_psi_2 = np.sin( (latp - lat)/2 )**2 + np.cos(latp) * np.cos(lat) * np.sin( (lonp - lon)/2 )**2 
    
    # Calculate Stokes' function S (Note 5)
    S = 1/np.sqrt(sin2_psi_2) - 6*np.sqrt(sin2_psi_2) + 1 - 5*cos_psi - \
        3*cos_psi*np.log(np.sqrt(sin2_psi_2) + sin2_psi_2)
        
    return S, cos_psi
    
def meissl(comp_point, int_points, psi_0) -> float:
    '''
    Calculate the Original Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    psi_0      : spherical distance of the spherical cap in radians
    
    Returns
    -------
    S         : Stokes' function
    '''
    S, _ = stokes(comp_point, int_points)
    S_0, _ = stokes([0, np.degrees(psi_0)], np.array([[0, 0]]))

    return S - S_0

def wong_and_gore(comp_point, int_points, nmax) -> float:
    '''
    Wong and Gore's modification of Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    nmax       : Maximum degree of expansion
    
    Returns
    -------
    S_wg      : Modified Stokes' function
    
    Notes
    -----
    1. Featherstone (2002): Software for computing five existing
       types of deterministically modified integration kernel for 
       gravimetric geoid determination
       https://www.sciencedirect.com/science/article/pii/S0098300402000742
    '''
    # Calculate original Stokes' function
    S, cos_psi = stokes(comp_point, int_points)
    
    # Wong and Gore's modification (Featherstone (2002): Eq. 21)
    S_wg = np.zeros_like(cos_psi)
    for i, t in enumerate(cos_psi):
        Pn = legendre_poly(t=t, nmax=nmax)
        sum_term = 0
        for n in range(2, nmax + 1):
            sum_term += (2 * n + 1) / (n - 1) * Pn[n]
        S_wg[i] = S[i] - sum_term
    
    return S_wg

def heck_and_gruninger(comp_point, int_points, psi_0, nmax) -> float:
    '''
    Heck and Gruninger's modification of Stokes' function
    
    Parameters
    ----------
    comp_point : array-like, shape (2,)
                    [lon, lat] of computation point in degrees
    int_points : array-like, shape (n, 2)
                    [lon, lat] of integration points in degrees
    psi_0      : spherical distance of the spherical cap in radians
    nmax       : Maximum degree of expansion
    
    Returns
    -------
    S_hg      : Heck and Gruninger's modification of Stokes' function
    
    Notes
    -----
    1. Featherstone (2002): Software for computing five existing
       types of deterministically modified integration kernel for 
       gravimetric geoid determination
       https://www.sciencedirect.com/science/article/pii/S0098300402000742
    '''
    # Calculate original Stokes' function
    # S, cos_psi = stokes(comp_point, int_points)
    
    # Wong and Gore
    S_wg = wong_and_gore(comp_point, int_points, nmax)
    
    # Featherstone (2002): Eq. 26
    # Stokes' function for a spherical cap (psi_0)
    S_0, cos_psi_0 = stokes([0, np.degrees(psi_0)], np.array([[0, 0]]))
    # Wong and Gore for spherical cap (psi_0)
    t = cos_psi_0
    Pn = legendre_poly(t=t, nmax=nmax)
    S_wgL = 0
    for n in range(2, nmax + 1):
        S_wgL += (2 * n + 1) / (n - 1) * Pn[n]
    
    # Heck and Gruninger
    S_hg = S_wg - (S_0 - S_wgL)
    
    return S_hg


def stokes_psi(psi) -> tuple:
    '''
    Calculate the Original Stokes' function from spherical distance
    
    Parameters
    ----------
    psi       : array-like
                    Spherical distance in radians
    
    Returns
    -------
    S         : Stokes' function
    cos_psi   : Cosine of spherical distance
    '''
    cos_psi = np.cos(psi)
    
    # Calculate sin^2(psi/2) using the relationship sin^2(psi/2) = (1 - cos(psi)) / 2
    sin2_psi_2 = (1 - cos_psi) / 2
    
    # Calculate Stokes' function S
    S = 1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - \
        3 * cos_psi * np.log(np.sqrt(sin2_psi_2) + sin2_psi_2)
    
    return S, cos_psi

def wong_and_gore_psi(psi, nmax) -> float:
    '''
    Wong and Gore's modification of Stokes' function from spherical distance
    
    Parameters
    ----------
    psi       : array-like
                    Spherical distance in radians
    nmax      : Maximum degree of expansion
    
    Returns
    -------
    S_wg      : Modified Stokes' function
    '''
    # Calculate original Stokes' function
    S, cos_psi = stokes_psi(psi)
    
    # Wong and Gore's modification
    S_wg = np.zeros_like(cos_psi)
    for i, t in enumerate(cos_psi):
        Pn = legendre_poly(t=t, nmax=nmax)
        sum_term = 0
        for n in range(2, nmax + 1):
            sum_term += (2 * n + 1) / (n - 1) * Pn[n]
        S_wg[i] = S[i] - sum_term
    
    return S_wg

def heck_and_gruninger_psi(psi, psi0, nmax) -> float:
    '''
    Heck and Gruninger's modification of Stokes' function from spherical distance
    
    Parameters
    ----------
    psi       : array-like
                    Spherical distances in radians
    psi0      : float
                 Spherical distance of the spherical cap in radians
    nmax      : Maximum degree of expansion
    
    Returns
    -------
    S_hg      : Heck and Gruninger's modification of Stokes' function
    '''
    # Calculate Wong and Gore's modification for psi
    S_wg = wong_and_gore_psi(psi, nmax=nmax)
    S_wg0 = wong_and_gore_psi(np.array([psi0]), nmax=nmax)
    # Calculate original Stokes' function for psi0
    S_0, _ = stokes_psi(np.array([psi0]))
    
    # Heck and Gruninger's modification
    S_hg = S_wg - (S_0[0] - S_wg0[0])
    
    return S_hg

def meissl_psi(psi, psi0) -> float:
    '''
    Meissl's modification of Stokes function
    
    Parameters
    ----------
    psi       : array-like, shape (n,)
                    Spherical distance in radians
    psi0      : float
                    Spherical distance of the spherical cap in radians
    
    Returns
    -------
    S_m       : Modified Stokes' function
    '''
    S, _ = stokes_psi(psi)
    S_w0 = stokes_psi(np.array(psi0))[0]
    
    return S - S_w0

# # For plotting purposes
# def stokes_func(sph_dist):
#     '''
#     Stokes' function for a given spherical distance
    
#     Parameters 
#     ----------
#     sph_dist  : spherical distance
    
#     Returns
#     -------
#     S         : Stokes' function
    
#     Notes
#     ---------
#     1. Physical Geodesy (2nd Edition) Page 104, Equation 2–305
#     2. For numerical efficiency and accuracy, we will use a slightly modified form of Equation 2-305
#     '''    
#     S = 1/np.sin(sph_dist/2) - 6*np.sin(sph_dist/2) + 1 - 5*np.cos(sph_dist) - \
#         3*np.cos(sph_dist)*np.log(np.sin(sph_dist/2) + np.sin(sph_dist/2)**2)
    
#     return S
