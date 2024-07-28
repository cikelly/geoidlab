############################################################
# Utilities for spherical harmonic synthesis               #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import constants
from numpy import (
    zeros, sqrt
)
import copy

def read_icgem(icgem_file:str):
    """
    Read spherical harmonic coefficients from an ICGEM .gfc file.

    Parameters
    ----------
    icgem_file : str
        The path to the ICGEM .gfc file.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'a'       : The reference radius.
        - 'nmax'    : The maximum degree of expansion.
        - 'GM'      : The Earth's gravitational constant.
        - 'Cnm'     : A numpy array containing the cosine coefficients.
        - 'Snm'     : A numpy array containing the sine coefficients.
        - 'sCnm'    : A numpy array containing the formal cosine errors.
        - 'sSnm'    : A numpy array containing the formal sine errors.
        - 'tide_sys': The tide system used in the model.
    """
    with open(icgem_file, 'r') as f:
        data = f.readlines()
    
    ##### Read a, GM, nmax
    keys = {
        'earth_gravity_constant': float,
        'radius': float,
        'max_degree': int,
        'tide_system': str
    }

    values = {}
    for line in data:
        for key, type_ in keys.items():
            if key in line:
                values[key] = type_(line.split()[1])

    nmax = values.get('max_degree')

    ##### Read Cnm, Snm, sCnm, sSnm
    Cnm = zeros( (nmax+1, nmax+1) )
    Snm = zeros( (nmax+1, nmax+1) )
    sCnm = zeros( (nmax+1, nmax+1) )
    sSnm = zeros( (nmax+1, nmax+1) )
    for line in data:
        if line.strip().startswith('gfc'):
            line = line.split()
            n = int(line[1])
            m = int(line[2])
            Cnm[n,m]  = float(line[3])
            Snm[n,m]  = float(line[4]) 
            sCnm[n,m] = float(line[5]) 
            sSnm[n,m] = float(line[6])

    shc             = {}
    shc['a']        = values.get('radius')
    shc['nmax']     = nmax
    shc['GM']       = values.get('earth_gravity_constant')
    shc['Cnm']      = Cnm
    shc['Snm']      = Snm
    shc['sCnm']     = sCnm
    shc['sSnm']     = sSnm
    shc['tide_sys'] = values.get('tide_system')

    return shc


def replace_Cn0(Cnm, GM1, a1, ellipsoid='wgs84'):
    """
    Replace C20, C40, C60, C80, and C100 coefficients

    Parameters
    ----------
    Cnm       : Cosine coefficients
    ellipsoid : str: 'wgs84' or 'grs80'
    GM1       : Model Earth gravitational constant
    a1        : Model Earth radius (semi-major axis)

    Returns
    -------
    Cnm
    """

    ellipsoid = constants.wgs84() if ellipsoid.lower()=='wgs84' else constants.grs80()
    GM2  = ellipsoid['GM']
    a2   = ellipsoid['semi_major']
    C20  = ellipsoid['C20']
    C40  = ellipsoid['C40']
    C60  = ellipsoid['C60']
    C80  = ellipsoid['C80']
    C100 = ellipsoid['C100']

    Cnm[2,0]  = Cnm[2,0]  - GM2/GM1 * (a2/a1)**2  * C20
    Cnm[4,0]  = Cnm[4,0]  - GM2/GM1 * (a2/a1)**4  * C40
    Cnm[6,0]  = Cnm[6,0]  - GM2/GM1 * (a2/a1)**6  * C60
    Cnm[8,0]  = Cnm[8,0]  - GM2/GM1 * (a2/a1)**8  * C80
    Cnm[10,0] = Cnm[10,0] - GM2/GM1 * (a2/a1)**10 * C100

    return Cnm   

def degree_amplitude(shc:dict, ellipsoid='wgs84'):
    """
    Parameters
    ----------
    shc       : spherical harmonic coefficients (output of read_icgem)

    Returns
    -------
    var       : variance
    """
    shc1 = copy.deepcopy(shc)
    # Update Cnm
    shc1['Cnm'] = replace_Cn0(shc1['Cnm'], shc1['GM'], shc1['a'], ellipsoid=ellipsoid)
    ellipsoid = constants.wgs84() if ellipsoid.lower()=='wgs84' else constants.grs80()
    
    coefficients  = [['Cnm', 'Snm'], ['sCnm', 'sSnm']]
    variance_dict = {}
    
    
    
    for i in range(len(coefficients)):
        geoid   = zeros(shc1['nmax']+1)
        degree  = zeros(shc1['nmax']+1)
        anomaly = zeros(shc1['nmax']+1)
        
        C = shc1[coefficients[i][0]]
        S = shc1[coefficients[i][1]]
        
        C2 = C**2
        S2 = S**2

        for n in range(1, shc1['nmax']+1):
            sum = 0
            for m in range(n+1):
                sum += C2[n, m] + S2[n, m]
            
            geoid[n]   = sqrt(shc1['a']**2 * sum)
            anomaly[n] = sqrt((shc1['GM'] / shc1['a']**2)**2 * 10**10 * (n-1)**2 * sum)
            
            degree[n]  = n
        
        # Assign to the appropriate keys in the variance_dict
        if i == 0:
            variance_dict['geoid'] = geoid
            variance_dict['anomaly'] = anomaly
        else:
            variance_dict['error_geoid'] = geoid
            variance_dict['error_anomaly'] = anomaly

    variance_dict['degree'] = degree

    return variance_dict  