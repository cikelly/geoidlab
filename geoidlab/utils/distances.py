############################################################
# Utilities for calculating distances                      #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np
from numba import njit, vectorize

def haversine(lon1, lat1, lon2, lat2, r=6371.0, unit='deg') -> float:
    '''
    Calculate the great-circle distance between two points on Earth in kilometers.
    
    Parameters
    ----------
    lon1      : longitude of first point in degrees
    lat1      : latitude of first point in degrees
    lon2      : longitude of second point in degrees
    lat2      : latitude of second point in degrees
    r         : Earth's radius in kilometers (default: 6371.0)
    unit      : unit of distance to return. Options are:
                    'deg' for degrees (default)
                    'rad' for radians
                    'km' for kilometers
                    'm' for meters
    
    Returns
    -------
    distance  : distance between the two points in kilometers 
    '''
    
    UNIT_FACTORS = {
        'rad': 1.0,
        'deg': 180.0 / np.pi,
        'km': r,
        'm': r * 1000.0
    }
    
    if unit not in UNIT_FACTORS:
        raise ValueError(f'Unit must be one of {list(UNIT_FACTORS.keys())}')

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula
    a = (
        np.sin(dlat / 2.0) ** 2 + 
        np.cos(lat1) * np.cos(lat2) * 
        np.sin(dlon / 2.0) ** 2
    )
    # c = 2 * np.arcsin(np.sqrt(a))
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a)) # more stable numerically
    
    # Convert to desired unit
    distance = c * UNIT_FACTORS[unit.lower()]
    
    return distance


@njit
def haversine_fast(lon1, lat1, lon2, lat2, in_unit='rad', out_unit='deg') -> float:
    '''
    Numba-optimized haversine function.
    
    Parameters
    ----------
    lon1      : longitude of first point
    lat1      : latitude of first point
    lon2      : longitude of second point
    lat2      : latitude of second point
    in_unit   : unit of input coordinates
    out_unit  : unit of distance to return
    
    Returns
    -------
    unit      : unit of distance to return
    '''
    if in_unit == 'deg':
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)
        
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2) ** 2 + 
        np.cos(lat1) * np.cos(lat2) * 
        np.sin(dlon / 2) ** 2
    )
    # c = 2 * np.arcsin(np.sqrt(a))
    c = 2 * np.atan2( np.sqrt(a), np.sqrt(1 - a) ) # more stable numerically
    
    return np.degrees(c) if out_unit == 'deg' else c

@vectorize
def haversine_vectorized(lon1, lat1, lon2, lat2, in_unit='rad', out_unit='deg') -> float:
    '''
    Vectorized haversine function.
    '''
    return haversine_fast(lon1, lat1, lon2, lat2, in_unit, out_unit)