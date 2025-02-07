############################################################
# Utilities for modeling potential change of the regular   #
# part of topography                                       #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np

class PotentialChange:
    def __init__(self, psi, h_p=None, rho=2.67, k=6.673e-11, R=6371, step=10) -> None:
        '''
        Parameters
        ----------
        psi       : angular radius [deg]
        h_p       : height [m]
        rho       : density [g/cm3]
        k         : gravitational constant [m3/kg/s]
        R         : earth radius [km]
        '''
        self.psi  = np.radians(psi)
        self.h_p  = h_p
        self.rho  = rho * 1e3             # density [kg/m3]
        self.k    = k
        self.R    = R * 1e3               # earth radius [m]
        # self.step = step
        
        if h_p is None:
            self.h_p = np.arange(0, 9000+step, step)
            
    def circular_plate(self) -> np.array:
        a = 2 * self.R * np.sin(self.psi / 2)
        E1 = np.sqrt(a**2 + self.h_p**2)
        V_prime = np.pi * self.k * self.rho * (-self.h_p**2 + self.h_p * E1 + a**2 * np.log( (self.h_p + E1) / a))
        Vs = 4 * np.pi * self.k * self.rho * self.h_p * self.R * np.sin(self.psi/2)
        
        return (V_prime - Vs) / 10
    
    def grushinsky(self) -> np.array:
        '''
        Grunshinsky's method for computing potential change at P_0 on the geoid
        '''
        return -(np.pi * self.k * self.rho * self.h_p**2) / 10
    
    
    def spherical_bouguer() -> np.array:
        pass
    
    # def plot_potential(self, xlim=None, ylim=None, key='circular_plate', ) -> None:
    #     # Titles corresponding to the keys
    #     titles = {
    #         'circular_plate': r'Potential Change at $P_0$ of a Circular Plate',
    #         'grushinsky': r"Potential Change at $P_0$ using Grushinsky's Method"
    #     }
        
    #     # Method corresponding to the key
    #     methods = {
    #         'circular_plate': self.circular_plate,
    #         'grushinsky': self.grushinsky
    #     }
        
    #     # Validate the key
    #     if key not in methods:
    #         raise ValueError(f"Invalid key '{key}'. Valid keys are: {list(methods.keys())}")
        
    #     # Select the appropriate method and title
    #     method = methods[key]
    #     title = titles[key]
        
    #     # Plotting
    #     plt.figure(figsize=[8, 9])
    #     plt.plot(self.h_p, method(), linewidth=0.8, color='black')
    #     plt.xlim([0, 10000]) if xlim is None else plt.xlim(xlim)
    #     plt.ylim([-5, 0]) if ylim is None else plt.ylim(ylim)
    #     plt.xlabel('Height [m]', fontsize=13.5)
    #     plt.ylabel('Potential Change [kgal-m]', fontsize=13.5)
    #     plt.grid(which='both', linewidth=0.5)
    #     plt.minorticks_on()
    #     plt.grid(which='minor', linewidth=0.25)
    #     plt.title(title, fontweight='bold', fontsize=16)
    #     plt.show()
