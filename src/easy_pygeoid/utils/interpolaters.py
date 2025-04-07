############################################################
# Utilities for interpolating data                         #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def clean_data(df, key='Dg') -> pd.DataFrame:
    '''
    Clean the input DataFrame (dropping NaNs and averaging duplicates).
    
    Parameters
    ----------
    df        : the input DataFrame
        
    
    Returns
    -------
    df_clean  : the cleaned DataFrame
    '''
    
    df_clean = df.dropna(subset=[key]).copy()
    df_clean = df_clean.groupby(['lon', 'lat'], as_index=False)[key].mean()
    return df_clean[['lon', 'lat', key]]

def scatteredInterpolant(
    df, 
    grid_extent, 
    resolution, 
    resolution_unit='minutes',
    data_key='Dg'
) -> np.ndarray:
    '''
    Interpolate scattered data using Delaunay triangulation and linear interpolation.

    Parameters
    ----------
    df             : the input DataFrame
    grid_extent    : the extent of the grid (lon_min, lon_max, lat_min, lat_max)
    resolution     : the resolution of the grid (in degrees or minutes)
    resolution_unit: unit of resolution ('degrees' or 'minutes')

    Returns
    -------
    grid           : the interpolated grid
    '''

    # Clean the data
    df_clean = clean_data(df, key=data_key)

    # Convert resolution to degrees if in minutes
    if resolution_unit == 'minutes':
        resolution = resolution / 60.0
    elif resolution_unit == 'seconds':
        resolution = resolution / 3600.0
    elif resolution_unit == 'degrees':
        pass
    else:
        raise ValueError('resolution_unit must be \'degrees\', \'minutes\', or \'seconds\'')


    # Create a grid for interpolation
    lon_min, lon_max, lat_min, lat_max = grid_extent
    lon_grid = np.arange(lon_min, lon_max + resolution, resolution)
    lat_grid = np.arange(lat_min, lat_max + resolution, resolution)

    # Create a meshgrid for interpolation
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)

    # Create Delaunay triangulation
    points = df_clean[['lon', 'lat']].values
    tri = Delaunay(points, qhull_options='Qt Qbb Qc Qz')

    values = df_clean[data_key].values
    interpolator = LinearNDInterpolator(tri, values)

    # Nearest neighbor extrapolation (unchanged)
    neighbor_interp = NearestNDInterpolator(points, values)

    data_linear = interpolator(np.column_stack((Lon.ravel(), Lat.ravel()))).reshape(Lon.shape)
    data_nearest = neighbor_interp(np.column_stack((Lon.ravel(), Lat.ravel()))).reshape(Lon.shape)

    data_interp = np.where(np.isnan(data_linear), data_nearest, data_linear)

    return Lon, Lat, data_interp.reshape(Lon.shape)
