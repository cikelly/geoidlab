###########################################################
#Utilities for spherical harmonic synthesis               #
#Copyright (c) 2024, Caleb Kelly                          #
#Author: Caleb Kelly  (2024)                              #
###########################################################
import constants

import numpy as np
import copy

def degree_amplitude(shc:dict, ellipsoid='wgs84', replace_zonal=True) -> dict:
    '''
    Calculate geoid and anomaly degree amplitude from spherical harmonic coefficients
    
    Parameters
    ----------
    shc       : spherical harmonic coefficients (output of read_icgem)

    Returns
    -------
    var       : variance
    '''
    shc1 = copy.deepcopy(shc)
    if replace_zonal:
        shc1 = subtract_zonal_harmonics(shc1, ellipsoid=ellipsoid)
    ellipsoid = constants.wgs84() if ellipsoid.lower()=='wgs84' else constants.grs80()
    
    coefficients  = [['Cnm', 'Snm'], ['sCnm', 'sSnm']]
    variance_dict = {}
    
    # for i in range(len(coefficients)):
    geoid   = np.zeros(shc1['nmax']+1)
    degree  = np.zeros(shc1['nmax']+1)
    anomaly = np.zeros(shc1['nmax']+1)
    
    C = shc1['Cnm']
    S = shc1['Snm']
    
    C2 = C ** 2
    S2 = S ** 2

    for n in range(1, shc1['nmax']+1):
        sum = 0
        for m in range(n+1):
            sum += C2[n, m] + S2[n, m]
        
        geoid[n]   = np.sqrt(shc1['a']**2 * sum)
        anomaly[n] = np.sqrt((shc1['GM'] / shc1['a']**2)**2 * 10**10 * (n-1)**2 * sum)
        
        degree[n]  = n
    
    variance_dict['geoid'] = geoid
    variance_dict['anomaly'] = anomaly
    variance_dict['degree'] = degree

    return variance_dict  

def error_degree_amplitude(shc:dict, ellipsoid='wgs84', replace_zonal=True) -> dict:
    '''
    Calculate geoid and anomaly error degree amplitude from spherical harmonic coefficients
    
    Parameters
    ----------
    shc                      : spherical harmonic coefficients (output of read_icgem)
    subtract_zonal_harmonics : Replace zonal harmonic coefficients (C[n,0])
    
    Returns
    -------
    variances               : Cumulative error variance (geoid and anomaly)
    
    Notes
    -----
    1. Torge, Müller & Pail (2023): Geodesy, 5 Edition
            (a) Gravity anomaly error: (P. 341, Eq. 6.138)
            (b) Geoid error: (P. 342, Eq. 6.139)
    '''
    shc1 = copy.deepcopy(shc)
    
    if replace_zonal:
        shc1 = subtract_zonal_harmonics(shc1, ellipsoid=ellipsoid)
    ellipsoid = constants.wgs84() if ellipsoid.lower()=='wgs84' else constants.grs80()
    
    variance_dict = {}
    
    geoid   = np.zeros(shc1['nmax']+1)
    degree  = np.zeros(shc1['nmax']+1)
    anomaly  = np.zeros(shc1['nmax']+1)
    
    dC = shc1['sCnm']
    dS = shc1['sSnm']
    
    dC2 = dC ** 2
    dS2 = dS ** 2

    sum1 = 0
    sum2 = 0
    for n in range(1, shc1['nmax']+1):
        sum = 0
        for m in range(n+1):
            sum += dC2[n, m] + dS2[n, m]
        
        sum1 += ((shc1['GM'] / shc1['a']**2)**2 * 10**10 * (n-1)**2)*sum # anomaly
        sum2 += shc1['a']**2 * sum                                       # geoid (m)
        anomaly[n] = sum1
        geoid[n]   = sum2
        degree[n]  = n
        
    anomaly = np.sqrt(anomaly)
    geoid   = np.sqrt(geoid)
    
    variance_dict['error_anomaly'] = anomaly
    variance_dict['error_geoid']   = geoid * 100 # geoid (cm)

    variance_dict['degree'] = degree

    return variance_dict 


def subtract_zonal_harmonics(
        shc: dict[str, np.ndarray],
        ellipsoid: str = 'wgs84'
    ) -> dict[str, np.ndarray]:
    '''
    Replace C20, C40, C60, C80, and C100 coefficients (zonal harmonics)

    Parameters
    ----------
    shc       : dict[str, numpy.np.ndarray]
                    spherical harmonic coefficients (out of icgem.read_icgem())
    ellipsoid : str
                    reference ellipsoid ('wgs84' or 'grs80')

    Returns
    -------
    shc       : dict[str, numpy.np.ndarray]
                    shc with updated Cn0 coefficients
    '''
    # Check if shc is not None
    if shc is None:
        raise ValueError('shc cannot be None')
    
    # Check if 'Cnm' and 'a' keys are in the dictionary
    if 'Cnm' not in shc or 'a' not in shc:
        raise KeyError('shc must have "Cnm" and "a" keys')
    
    # Reference ellipsoid: remove the even zonal harmonics from sine and cosine coefficients.
    ref_ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
    GMe = ref_ellipsoid['GM']
    a_e = ref_ellipsoid['semi_major']

    zonal_harmonics = ['C20', 'C40', 'C60', 'C80', 'C100']

    for n, Cn0 in zip([2, 4, 6, 8, 10], zonal_harmonics):
        if Cn0 not in ref_ellipsoid:
            raise KeyError(f"{Cn0} coefficient not found in the reference ellipsoid")
        shc['Cnm'][n, 0] = shc['Cnm'][n, 0] - (GMe / shc['GM']) * (a_e / shc['a']) ** n * ref_ellipsoid[Cn0]

    return shc

