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

from mpl_toolkits.basemap import Basemap

def plot_gravity_anomaly(
    data=None, gravity_data=None,
    which='both', colormap='jet',
    ellipsoid='wgs84', interp='nearest',
    figsize=(12, 6), save=False, step=1,
    origin='lower', plot_interp=None,
    vmin=None, vmax=None
) -> None:
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
        print('Calculating gravity anomalies ...')
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
    
    # points = np.column_stack((lon, lat))
    step = km2deg(step)
    Lon = np.arange(lon.min(), lon.max()+step, step)
    Lat = np.arange(lat.min(), lat.max()+step, step)
    
    Lon, Lat = np.meshgrid(Lon,Lat)

    # Interpolate
    freeAir = scipy.interpolate.Rbf(lon, lat, free_air_anomaly, function='thin_plate')
    bouguer_G = scipy.interpolate.Rbf(lon, lat, bouguer_anomaly, function='thin_plate')
    
    freeAir = freeAir(Lon, Lat)
    bouguer_G = bouguer_G(Lon, Lat)
    
    # Make maps
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    if which == 'both':
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        
        # im = axs[0].imshow(freeAir, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp, vmin=vmin, vmax=vmax)
        im = axs[0].pcolormesh(Lon, Lat, freeAir, cmap=colormap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[0], label='Gravity anomaly (mGal)', aspect=25, pad=0.03)
        
        # im = axs[1].imshow(bouguer_G, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp, vmin=vmin, vmax=vmax)
        im = axs[1].pcolormesh(Lon, Lat, bouguer_G, cmap=colormap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[1], label='Gravity anomaly (mGal)', aspect=25, pad=0.03)
        titles = ['Free-air', 'Bouguer']
        _ = [axs[i].set_title(titles[i], fontweight='bold') for i in range(2)]
        fig.tight_layout()
        if save:
            plt.savefig('gravity_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
    elif which == 'free_air':
        plt.imshow(freeAir, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp, vmin=vmin, vmax=vmax)
        plt.title('Free-air', fontweight='bold')
        plt.colorbar(label='Gravity anomaly (mGal)', aspect=25, pad=0.03)
        if save:
            plt.savefig('gravity_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.imshow(bouguer_G, cmap=colormap, extent=extent, origin=origin, interpolation=plot_interp, vmin=vmin, vmax=vmax)
        plt.title('Bouguer', fontweight='bold')
        plt.colorbar(label='Gravity anomaly (mGal)', aspect=25, pad=0.03)
        if save:
            plt.savefig('gravity_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    # return free_air_anomaly
    
def km2deg(km) -> float:
    '''
    Convert km to degrees
    '''
    return km / 111.11

def plot_degree_variances() -> None:
    '''
    
    '''
    pass


def drape_over_topography(
    lons, 
    lats, 
    values, 
    topography, 
    cmap='RdBu', 
    vmin=None, 
    vmax=None, 
    figsize=(10,5)
) -> None:
    """
    Plot variable values over a topography map.

    Parameters:
    -----------
    lons : 1D array
        Longitude coordinates of values.
    lats : 1D array
        Latitude coordinates of values.
    values : 1D array
        Values to be plotted.
    topography : 2D array
        Topography data.
    cmap : str or colormap, optional, default: 'RdBu'
        Colormap to use.
    vmin : float, optional, default: None
        Minimum value for colorbar.
    vmax : float, optional, default: None
        Maximum value for colorbar.
    figsize : tuple, optional, default: (10,5)
        Figure size.

    Returns:
    --------
    None
    """
    # Make a Basemap object
    m = Basemap(projection='cyl', llcrnrlat=lats.min(), urcrnrlat=lats.max(),
                llcrnrlon=lons.min(), urcrnrlon=lons.max(), resolution='c')

    # Calculate the grid coordinates
    x, y = m(lons, lats)

    # Create a mask for the topography
    mask = np.isnan(topography)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the topography
    m.imshow(topography, cmap='terrain', interpolation='nearest',
             extent=(lons.min(), lons.max(), lats.min(), lats.max()))

    # Plot the values
    im = m.pcolor(x, y, values, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)

    # Set the colorbar
    cbar = m.colorbar(im, location='right', pad='10%')

    # Set the title
    ax.set_title('Variable values over topography')

    # Set the extent of the plot
    m.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    # Set the masked area to white
    m.drawmapboundary(fill_color='white')

    # Draw the coastlines
    m.drawcoastlines()

    # Draw the parallels
    m.drawparallels(np.arange(-90, 91, 30))

    # Draw the meridians
    m.drawmeridians(np.arange(-180, 181, 60))

    # Show the plot
    plt.show()

# TODO: Add function to drape variable over topography



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.mplot3d import Axes3D

# # Load topographic data (example data)
# topo_data = np.load('topo_data.npy')  # Replace with actual data
# geoid_data = np.load('geoid_data.npy')  # Replace with actual data

# # Create a map with Gall stereographic cylindrical projection[^2^][2]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# m = Basemap(projection='gall', lon_0=0, ax=ax)

# # Create meshgrid for topographic and geoid data
# lon = np.linspace(-180, 180, topo_data.shape[1])
# lat = np.linspace(-90, 90, topo_data.shape[0])
# lon, lat = np.meshgrid(lon, lat)

# # Plot geoid data with topographic relief[^1^][1]
# m.plot_surface(lon, lat, topo_data, facecolors=plt.cm.viridis(geoid_data), rstride=1, cstride=1, antialiased=True)

# # Add coastlines for reference
# m.drawcoastlines()

# # Set view angle
# ax.view_init(elev=30, azim=120)

# plt.show()
