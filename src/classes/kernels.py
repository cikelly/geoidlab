import numpy as np
from legendre import legendre_poly

class StokesCalculator:
    def __init__(self, comp_point, int_points):
        self.comp_point = np.array(comp_point)
        self.int_points = np.array(int_points) if isinstance(int_points, list) else int_points
        self.lon, self.lat = self.int_points[:, 0], self.int_points[:, 1]
        self.lonp, self.latp = np.radians(self.comp_point)
        self.lon, self.lat = np.radians(self.lon), np.radians(self.lat)
    
    def stokes(self):
        cos_dlam = np.cos(self.lon) * np.cos(self.lonp) + np.sin(self.lon) * np.sin(self.lonp)
        cos_psi = np.sin(self.latp) * np.sin(self.lat) + np.cos(self.latp) * np.cos(self.lat) * cos_dlam
        sin2_psi_2 = np.sin((self.latp - self.lat) / 2) ** 2 + np.cos(self.latp) * np.cos(self.lat) * np.sin((self.lonp - self.lon) / 2) ** 2
        S = 1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - 3 * cos_psi * np.log(np.sqrt(sin2_psi_2) + sin2_psi_2)
        return S, cos_psi
    
    def meissl(self, psi_0):
        S, _ = self.stokes()
        S_0, _ = self.stokes([0, np.degrees(psi_0)], np.array([[0, 0]]))
        return S - S_0
    
    def wong_and_gore(self, nmax):
        S, cos_psi = self.stokes()
        S_wg = np.zeros_like(cos_psi)
        for i, t in enumerate(cos_psi):
            Pn = legendre_poly(t=t, nmax=nmax)
            sum_term = 0
            for n in range(2, nmax + 1):
                sum_term += (2 * n + 1) / (n - 1) * Pn[n]
            S_wg[i] = S[i] - sum_term
        return S_wg
    
    def heck_and_gruninger(self, psi_0, nmax):
        S_wg = self.wong_and_gore(nmax)
        S_0, cos_psi_0 = self.stokes([0, np.degrees(psi_0)], np.array([[0, 0]]))
        t = cos_psi_0
        Pn = legendre_poly(t=t, nmax=nmax)
        S_wgL = 0
        for n in range(2, nmax + 1):
            S_wgL += (2 * n + 1) / (n - 1) * Pn[n]
        S_hg = S_wg - (S_0 - S_wgL)
        return S_hg
