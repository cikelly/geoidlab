############################################################
# Utilities for automatic DEM download                     #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import requests
import os
import sys
import warnings
import netCDF4
import xarray as xr
import numpy as np
from tqdm import tqdm

warnings.simplefilter('ignore')

def get_readme_path():
    '''
    Function to get the path of README.V11.txt, which is required for automatic DEM download.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    readme_path   : absolute path of the README.V11.txt
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.join(script_dir, '../src/data/README.V11.txt')
    
    return os.path.abspath(readme_path)

def download_srtm30plus(url=None, downloads_dir=None, bbox=None):
    '''
    Download SRTM30PLUS from https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/
    
    Parameters
    ----------
    url           : url of the srtm30plus tile
    downloads_dir : directory to download the file to
    bbox          : bbox of the area of interest (to be provided if url is not provided)
    
    Returns
    -------
    fname         : Name of downloaded file
    '''
    if not url:
        url = fetch_url(bbox=bbox)
        
    filename = url.split('/')[-1]
    filepath = os.path.join(downloads_dir, filename) if downloads_dir else filename
    
    # Check if the file already exists
    if os.path.exists(filepath):
        try:
            response_head = requests.head(url, verify=False)
            total_size = int(response_head.headers.get('content-length', 0))
            # Check if the existing file size matches the expected size
            if os.path.getsize(filepath) == total_size:
                print(f'File {filename} already exists and is complete. Skip download\n')
                return filename
            else:
                print(f'{filename} already exists but is incomplete. Redownloading ...\n')
                os.remove(filepath)
        except requests.exceptions.RequestException:
            print(f'Unable to check if {filename} is complete. {filename} in {downloads_dir} will be used ...\n')
            return filename
            
    if downloads_dir:
        os.makedirs(downloads_dir, exist_ok=True)
        print(f'Downloading {filename} to {downloads_dir} ...\n')
    else:
        print(f'Downloading {filename} to {os.getcwd()} ...\n')
    
    # Download NetCDF file
    try:
        response = requests.get(url, verify=False, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(
            # desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            dynamic_ncols=True
        ) as pbar:
            
            with open(filepath, 'wb') as f:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(len(data))
                    pbar.refresh()
                    f.flush()
                    sys.stdout.flush()
    except Exception as e:
            raise RuntimeError(f'Download failed: {e}. Are you connected to the internet?')
                    
    return filename


def fetch_url(bbox):
    '''
    Get the url of the srtm30plus tiles
    
    Parameters
    ----------
    bbox          : bounding box of the area to download
                        [min_lon, min_lat, max_lon, max_lat]
                        [left, bottom, right, top]
    downloads_dir : directory to download the file to
    
    Returns
    -------
    url           : url of the srtm30plus tile
    '''
    readme_path = get_readme_path()
    # Define the base URL
    base_url = 'https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/'

    # Read the tile boundaries from the README file
    tiles = {}
    with open(readme_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5 and parts[0][0] in 'we':
                tile = parts[0]
                lat_min, lat_max = map(int, parts[1:3])
                lon_min, lon_max = map(int, parts[3:5])
                tiles[tile] = {'lon': (lon_min, lon_max), 'lat': (lat_min, lat_max)}

    # Find the tile that contains the bounding box
    for tile, bounds in tiles.items():
        if (bounds['lon'][0] <= bbox[0] < bounds['lon'][1] and
            bounds['lat'][0] <= bbox[1] < bounds['lat'][1]):
            return base_url + tile + '.nc'

    raise ValueError('No tile found that contains the bounding box')


def dem4geoid(bbox, ncfile=None, bbox_off=2, downloads_dir=None):
    '''
    Prepare a DEM for geoid calculation.
    
    Parameters
    ----------
    bbox          : bounding box of area of interest 
                    [xmin, ymin, xmax, ymax]
                    [left, bottom, right, top]
    ncfile        : path to DEM netCDF file
    bbox_off      : offset for bounding box (in degrees)
    downloads_dir : directory to download the file to
    
    Returns
    -------
    dem       : xarray dataset of the DEM
    
    Notes
    -----
    1. If DEM is already downloaded to file, either:
        a. specify both
            i. ncfile = filename with extension (without path); 
           ii. downloads_dir = directory where the ncfile is located
        b. specify only ncfile (full path)
    '''
    if not bbox:
        raise ValueError('Bounding box must be provided')
    
    if not ncfile:
        ncfile = download_srtm30plus(bbox=bbox, downloads_dir=downloads_dir)
        
    filepath = os.path.join(downloads_dir, ncfile) if downloads_dir else ncfile
    
    print(f'\nCreating xarray dataset of DEM over area of interest with buffer of {bbox_off} degrees\n')    
    
    nc = netCDF4.Dataset(filepath)
    fill_value = nc.variables['z']._FillValue
    
    ds = xr.open_dataset(filepath)
    ds['z']  = ds['z'].where(ds['z'] != fill_value, np.nan)
    
    # Subset over bbox
    bbox_subset = [bbox[0] - bbox_off, bbox[1] - bbox_off, bbox[2] + bbox_off, bbox[3] + bbox_off]
    dem = ds.sel(x=slice(bbox_subset[0], bbox_subset[2]), y=slice(bbox_subset[1], bbox_subset[3]))
    
    print('DEM created successfully!\n')
    
    return dem
