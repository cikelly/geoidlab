import requests
import os
import warnings
import netCDF4
import xarray as xr
import numpy as np
from tqdm import tqdm

warnings.simplefilter('ignore')

def download_srtm30plus(url=None, downloads_dir=None, bbox=None):
    '''
    
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
        response_head = requests.head(url, verify=False)
        total_size = int(response_head.headers.get('content-length', 0))

        # Check if the existing file size matches the expected size
        if os.path.getsize(filepath) == total_size:
            print(f'File {filename} already exists and is complete. Skip download\n')
            return filename
        else:
            print(f'File {filename} already exists but is incomplete. Redownloading ...\n')
            os.remove(filepath)
            
    if downloads_dir:
        os.makedirs(downloads_dir, exist_ok=True)
        # os.chdir(downloads_dir)
        print(f'Downloading {os.path.join(downloads_dir, filename)} to {downloads_dir} ...')
    else:
        print(f'Downloading {filename} to {os.getcwd()} ...')
        
    response = requests.get(url, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
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

    # Define the base URL
    base_url = 'https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/'

    # Read the tile boundaries from the README file
    tiles = {}
    with open('../src/data/README.V11.txt', 'r') as f:
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
    
    return dem