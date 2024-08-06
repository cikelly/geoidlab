############################################################
# Constants for gravity field modelling                    #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
def grs80():
    """
    GRS 1980 reference ellipsoid parameters
    
    Parameters
    ----------

    Returns
    -------
    dict
        semi_major: semi-major axis (m)
        semi_minor: semi-minor axis (m)
        GM          : geocentric gravitational constant (m**3 s**-2)
        J2          : dynamical form factor of the Earth 
        w           : angular velocity of the earth (rad s**-1)
        E           : linear eccentricity
        c           : polar radius of curvature (m)
        e2          : first eccentricity squared
        ep2         : second eccentricity squared
        f           : flattening
        1/f         : reciprocal flattening
        U0          : normal potential at the ellipsoid (m**2 s**-2)
        J4          : spherical harmonic coefficient
        J6          : spherical harmonic coefficient
        J8          : spherical harmonic coefficient
        m           : w**2a**2b/(GM)
        gamma_a     : normal gravity at the equator (m s**-2)
        gamma_b     : normal gravity at the pole (m s**-2)
    """

    grs80 = {
        'semi_major': 6_378_137,
        'semi_minor': 6_356_752.3141,
        'GM'        : 3_986_005e8,
        'J2'        : 108_263e-8,
        'w'         : 7_292_115e-11,
        'E'         : 521_854.0097,
        'c'         : 6_399_593.6259,
        'e2'        : 0.00669438002290, 
        'ep2'       : 0.00673949677548,
        'f'         : 0.003352810681,
        '1/f'       : 298.257222101,
        'U0'        : 62_636_860.850,
        'J4'        : -0.00000237091222,
        'J6'        : 0.00000000608347,
        'J8'        : -0.00000000001427,
        'm'         : 0.00344978600308,
        'gamma_a'   : 9.7803267715,
        'gamma_b'   : 9.8321863685,
        'C20'       : -0.484166854903603e-03,
        'C40'       : 0.790304072916597e-06,
        'C60'       : -0.168725117581045e-08,
        'C80'       : 0.346053239866698e-11,
        'C100'      : -0.265006218130312e-14,
    }
    return grs80



def wgs84():
    """
    WGS 1984 reference ellipsoid parameters
    
    Parameters
    ----------

    Returns
    -------
    dict
        semi_major  : semi-major axis (m)
        semi_minor  : semi-minor axis (m)
        GM          : geocentric gravitational constant (m**3 s**-2)
        w           : angular velocity of the earth (rad s**-1)
        E           : linear eccentricity
        c           : polar radius of curvature (m)
        e           : first eccentricity
        e2          : first eccentricity squared
        ep2         : second eccentricity squared
        f           : flattening
        U0          : normal potential at the ellipsoid (m**2 s**-2)
        m           : w**2a**2b/(GM)
        gamma_a     : normal gravity at the equator (m s**-2)
        gamma_b     : normal gravity at the pole (m s**-2)
    """

    wgs84 = {
        'semi_major': 6_378_137,
        'semi_minor': 6_356_752.3142,
        'GM'        : 3_986_004.418e8,
        'w'         : 7_292_115e-11,
        'E'         : 5.2185400842339e5,
        'c'         : 6_399_593.6258,
        'e'         : 8.1819190842622e-2,
        'e2'        : 6.69437999014e-3, 
        'ep'        : 8.2094437949696e-2,
        'ep2'       : 6.73949674228e-3,
        'f'         : 1/298.257223563,
        'U0'        : 62_636_851.7146,
        'm'         : 0.00344978650684,
        'gamma_a'   : 9.7803253359,
        'gamma_b'   : 9.8321849378,
        'mean_gamma': 9.7976432222,
        'C20'       : -0.484166774985e-03,
        'C40'       : 0.790303733511e-06,
        'C60'       : -0.168724961151e-08,
        'C80'       :  0.346052468394e-11,
        'C100'      : -0.265002225747e-14,
    }
    return wgs84
    
def earth():
    '''
    Constants for Earth/Geoid
    
    References
    ----------
    1. Gravity potential (W0) of the geoid:
        a. Sanchez et al. (2016), A conventional value for the geoid reference potential W0.
           Journal of Geodesy, 90, 815-835
        b. Sanchez & Sideris (2017), Vertical datum unification for the International Height Reference System (IHRS). 
           Geophysical journal international, 209, 570–586
    '''
    
    earth = {
        'W0'        : 62_636_853.40,  # Geoid potential (m2/s2)
        'radius'    : 6_371_000,      # Mean Earth radius (m)
    }
