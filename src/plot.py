############################################################
# Utilities for plotting                                   #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gravity as pygrav

import scipy.interpolate

def plot_gravity_anomaly(
    data=None, gravity_data=None,
    which='both', colormap='jet',
    ellipsoid='wgs84', interp='nearest',
    figsize=(12, 6), save=False, step=1,
    origin='lower', plot_interp=None
):
    '''
    Plot the free-air and Bouguer gravity anomaly
    
    Parameters
    ----------
    data         : data containing gravity anomalies 
                   columns of data should be: 
                        lon, lat, free_air, bouguer
    gravity_data : gravity data (if data is not provided)
                   gravity_data must contain (in order)
                        lon, lat, gravity, and elevation
    which        : plot 'both' or 'free_air' or 'bouguer'
    colormap     : colormap for the plot
    ellipsoid    : reference ellipsoid (wgs84 or grs80)
    interp       : interpolation method for gridding the data
    figsize      : size of the figure
    save         : save the figure
    step         : step size for gridding the data (km)
    origin       : 'lower' or 'upper'
    plot_interp  : Interpolation method for imshow
    
    Returns
    -------
    None
    '''
    if data is None and gravity_data is None:
        raise ValueError('data or gravity_data must be provided')
    
    # Get lon, lat, free_air_anomaly, and bouguer_anomaly
    if data is None:
        if isinstance(gravity_data, np.ndarray):
            free_air_anomaly, bouguer_anomaly = pygrav.gravity_anomalies(
                lat=gravity_data[:,1], 
                gravity=gravity_data[:,2], 
                elevation=gravity_data[:,3], 
                ellipsoid=ellipsoid
            )
            lon = gravity_data[:,0]
            lat = gravity_data[:,1]
        elif isinstance(gravity_data, pd.DataFrame):
            lon_column = [col for col in gravity_data.columns if pd.Series(col).str.contains('lon', case=False).any()][0]
            lat_column = [col for col in gravity_data.columns if pd.Series(col).str.contains('lat', case=False).any()][0]
            try:
                elev_column = [col for col in gravity_data.columns if pd.Series(col).str.contains('elev', case=False).any()][0]
            except IndexError:
                elev_column = [col for col in gravity_data.columns if pd.Series(col).str.contains('height', case=False).any()][0]
            grav_column = [col for col in gravity_data.columns if pd.Series(col).str.contains('grav', case=False).any()][0]
            
            free_air_anomaly, bouguer_anomaly = pygrav.gravity_anomalies(
                lat=gravity_data[lat_column],
                gravity=gravity_data[grav_column],
                elevation=gravity_data[elev_column],
                ellipsoid=ellipsoid
            )
            lon = gravity_data[lon_column]
            lat = gravity_data[lat_column]
    else:
        if isinstance(data, np.ndarray):
            lon, lat = data[:,0], data[:,1]
            free_air_anomaly, bouguer_anomaly = data[:,2], data[:,3]
        elif isinstance(data, pd.DataFrame):
            lon_column = [col for col in data.columns if pd.Series(col).str.contains('lon', case=False).any()][0]
            lat_column = [col for col in data.columns if pd.Series(col).str.contains('lat', case=False).any()][0]
            free_column = [col for col in data.columns if pd.Series(col).str.contains('free', case=False).any()][0]
            bouguer_column = [col for col in data.columns if pd.Series(col).str.contains('bouguer', case=False).any()][0]
            lon, lat = data[lon_column], data[lat_column]
            free_air_anomaly, bouguer_anomaly = data[free_column], data[bouguer_column]
    
    points = np.column_stack((lon, lat))
    step = km2deg(step)
    Lon = np.arange(lon.min(), lon.max()+step, step)
    Lat = np.arange(lat.min(), lat.max()+step, step)
    Lon, Lat = np.meshgrid(Lon,Lat)
    # Grid free_air_anomaly and bouguer_anomaly
    freeAir = scipy.interpolate.griddata(points, free_air_anomaly, (Lon, Lat), method=interp)
    bouguer_G = scipy.interpolate.griddata(points, bouguer_anomaly, (Lon, Lat), method=interp)
    
    # Make maps
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    if which == 'both':
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        
        im = axs[0].imshow(freeAir, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp)
        fig.colorbar(im, ax=axs[0], shrink=0.7, label='Gravity anomaly (mGal)')
        
        im = axs[1].imshow(bouguer_G, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp)
        fig.colorbar(im, ax=axs[1], shrink=0.7, label='Gravity anomaly (mGal)')
        titles = ['Free-air', 'Bouguer']
        _ = [axs[i].set_title(titles[i], fontweight='bold') for i in range(2)]
        fig.tight_layout()
        if save:
            plt.savefig('gravity_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
    elif which == 'free_air':
        plt.imshow(freeAir, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp)
        plt.title('Free-air', fontweight='bold')
        plt.colorbar(shrink=.95, label='Gravity anomaly (mGal)')
        if save:
            plt.savefig('gravity_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.imshow(bouguer_G, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp)
        plt.title('Bouguer', fontweight='bold')
        plt.colorbar(shrink=.95, label='Gravity anomaly (mGal)')
        if save:
            plt.savefig('gravity_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
        
def km2deg(km):
    '''
    Convert km to degrees
    '''
    return km / 111.11

def plot_degree_variances():
    '''
    
    '''
    pass