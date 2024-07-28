from numpy import (
    cos, sin, radians,
    sqrt, log
)

def stokes(lonp, latp, lon, lat):
    '''
    Original Stokes' kernel 
    
    Parameters
    ----------
    lonp      : longitude of computation point (in degrees)
    latp      : latitude of computation point (in degrees)
    lon       : longitude of running/integrating points (in degrees)
    lat       : latitude of running/integrating points (in degrees)
    
    Returns
    -------
    S         : Stokes' kernel
    sph_dist  : Spherical distance
    '''
    
    lonp = radians(lonp)
    latp = radians(latp)
    lon  = radians(lon)
    lat  = radians(lat)
    
def stokes_func(sph_dist):
    '''
    Stokes' function for a given spherical distance
    
    Parameters 
    ----------
    sph_dist  : spherical distance
    
    Returns
    -------
    S         : Stokes' function
    
    Notes
    ---------
    1. Physical Geodesy (2nd Edition) Page 104, Equation 2–305
    2. For numerical efficiency and accuracy, we will use a slightly modified form of Equation 2-305
    '''
    # S = 1 / sqrt(sph_dist) - 6 * sqrt(sph_dist) + 1 - 5 * cos(sph_dist) - \
    #     3 * cos(sph_dist) * log(sqrt(sph_dist) + sph_dist)
    
    S = 1/sin(sph_dist/2) - 6*sin(sph_dist/2) + 1 - 5*cos(sph_dist) - \
        3*cos(sph_dist)*log(sin(sph_dist/2) + sin(sph_dist/2)**2)
    
    return S

# def S_Stokes(lon, lat, lonp, latp):
#     """
#     Compute Stokes' function and the cosine of the spherical distance between points.
    
#     Parameters:
#     lon (array): Longitudes of integrating points.
#     lat (array): Latitudes of integrating points.
#     lonp (float): Longitude of the computation point.
#     latp (float): Latitude of the computation point.
    
#     Returns:
#     S (array): Stokes' function values.
#     cos_psi (array): Cosine of the spherical distance between points.
#     """
#     # Convert degrees to radians if necessary
#     lon = np.radians(lon)
#     lat = np.radians(lat)
#     lonp = np.radians(lonp)
#     latp = np.radians(latp)

#     # Cosine of the longitude difference
#     cos_dlam = np.cos(lon) * np.cos(lonp) + np.sin(lon) * np.sin(lonp)

#     # Cosine of the spherical distance (cosine of the central angle)
#     cos_psi = np.sin(latp) * np.sin(lat) + np.cos(latp) * np.cos(lat) * cos_dlam

#     # Auxiliary quantity (square of the sine of half the spherical distance)
#     sin2_psi_2 = np.sin((latp - lat) / 2) ** 2 + \
#                  np.sin((lonp - lon) / 2) ** 2 * \
#                  np.cos(latp) * np.cos(lat)

#     # Stokes's function (using the auxiliary quantity and the cosine of the spherical distance)
#     S = 1 / np.sqrt(sin2_psi_2) - 6 * np.sqrt(sin2_psi_2) + 1 - 5 * cos_psi - \
#         3 * cos_psi * np.log(np.sqrt(sin2_psi_2) + sin2_psi_2)

#     return S, cos_psi

# Example usage
# lon = np.array([0, 1, 2])
# lat = np.array([0, 1, 2])
# lonp = 0.5
# latp = 0.5

# S, cos_psi = S_Stokes(lon, lat, lonp, latp)
# print("Stokes' function values:", S)
# print("Cosine of spherical distances:", cos_psi)
    
    

def wong_and_gore():
    '''
    '''
    pass

def heck_and_gruninger():
    '''
    '''
    
    
