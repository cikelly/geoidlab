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
import rioxarray as rxr
import numpy as np
from tqdm import tqdm
from rasterio.enums import Resampling

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
    readme_path = os.path.join(script_dir, '../easy_pygeoid/data/README.V11.txt')
    
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
                print(f'{filename} already exists and is complete. Skip download\n')
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


def dem4geoid(
    bbox, ncfile=None, 
    bbox_off=1, downloads_dir=None,
    resolution=None
):
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
    resolution    : Resolution to resample the DEM
    
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
    
    2. 
    '''
    if not bbox:
        raise ValueError('Bounding box must be provided')
    
    if not ncfile:
        ncfile = download_srtm30plus(bbox=bbox, downloads_dir=downloads_dir)
        
    filepath = os.path.join(downloads_dir, ncfile) if downloads_dir else ncfile
    
    print(f'Creating xarray dataset of DEM with buffer of {bbox_off} degree(s)\n')    
    
    nc = netCDF4.Dataset(filepath)
    fill_value = nc.variables['z']._FillValue
    
    ds = xr.open_dataset(filepath)
    ds['z']  = ds['z'].where(ds['z'] != fill_value, np.nan)
    
    bbox_subset = [
        bbox[0] - bbox_off,
        bbox[1] - bbox_off,
        bbox[2] + bbox_off,
        bbox[3] + bbox_off
    ]
    # dem = ds.sel(
    #     x=slice(bbox_subset[0], bbox_subset[2]),
    #     y=slice(bbox_subset[1], bbox_subset[3])
    # )
    
    minx, maxx = bbox_subset[0], bbox_subset[2]
    miny, maxy = bbox_subset[1], bbox_subset[3]
    
    num_x_points = int((maxx - minx) / (resolution/3600)) + 1
    num_y_points = int((maxy - miny) / (resolution/3600)) + 1
    
    dem = ds.interp(
        x=np.linspace(minx, maxx, num_x_points),
        y=np.linspace(miny, maxy, num_y_points),
        method='nearest'
    )
    
    if dem.rio.crs is None:
        dem.rio.write_crs('EPSG:4326', inplace=True)
    
    if resolution and resolution != 30:
        print(f'Resampling DEM to {resolution} arc-seconds...')
        dem = dem.rio.reproject(dem.rio.crs, resolution=resolution/3600, resampling=Resampling.nearest)
    return dem

def download_dem_cog(
    bbox, model='None', cog_url=None,
    downloads_dir=None,
    bbox_off=2, resolution=30
):
    '''
    Download DEM using Cloud Optimized GeoTIFF (COG) format
    from OpenTopography.
    
    Parameters
    ----------
    bbox          : bbox of the area of interest (W, S, E, N)
    model         : name of the DEM model
                     - srtm
                     - cop
                     - nasadem
                     - gebco
    cog_url       : url of the COG
    downloads_dir : directory to download the file to
    bbox_off      : offset for bounding box (in degrees)
    resolution    : resolution to resample the DEM (in seconds)
    
    Returns
    -------
    dem           : xarray dataset of the DEM
    
    Notes
    -----
    1. OpenTopography hosts other DEMs that are not included here. 
    2. You can see the list of available COG URLs running the following command:
        `aws s3 ls s3://raster --recursive --endpoint-url https://opentopography.s3.sdsc.edu --no-sign-request > COG_urls.txt`
    3. References for `aws`:
        - https://aws.amazon.com/cli/
        - https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
    4. Unless you have a strong internet connection, I strongly recommend to use the default SRTM30PLUS
    '''
    models_url = {
        'srtm'   : 'https://opentopography.s3.sdsc.edu/raster/SRTM_GL3/SRTM_GL3_srtm.vrt',
        'cop'    : 'https://opentopography.s3.sdsc.edu/raster/COP90/COP90_hh.vrt',
        'nasadem': 'https://opentopography.s3.sdsc.edu/raster/NASADEM/NASADEM_be.vrt',
        'gebco'  : 'https://opentopography.s3.sdsc.edu/raster/GEBCOIceTopo/GEBCOIceTopo.vrt'
    }
    
    resolution = resolution / 3600 # convert seconds to degrees
    
    if model is not None and cog_url is None:
        try:
            cog_url = models_url[model.lower()]
        except KeyError:
            print('Supported models are:\nsrtm\ncop\netopo\nnasadem')
            raise ValueError(f'Unsupported model: {model}')
        
    # Next three conditions will use SRTM30Plus
    if model is None and cog_url is None:
        print('No model or COG URL provided. SRTM30Plus will be downloaded...\n')
        return dem4geoid(bbox=bbox, bbox_off=bbox_off, downloads_dir=downloads_dir, resolution=resolution*3600)
    
    if model.lower() == 'srtm' and resolution*3600 >= 30:
        print(f'You have requested SRTM at {resolution * 3600} arc-second resolution. SRTM30Plus will be downloaded...\n')
        dem = dem4geoid(bbox=bbox, bbox_off=bbox_off, downloads_dir=downloads_dir, resolution=resolution*3600)

        return dem
    
    if model is None and cog_url == models_url['srtm']:
        if resolution == 30/3600:
            print(f'You have requested 30 arc-second resolution. SRTM30Plus will be downloaded...\n')
            return dem4geoid(bbox=bbox, bbox_off=bbox_off, downloads_dir=downloads_dir, resolution=resolution*3600)
        else:
            print(f'You have requested {resolution} arc-second resolution. SRTM30Plus will be downloaded...\n')
            return dem4geoid(bbox=bbox, bbox_off=bbox_off, downloads_dir=downloads_dir, resolution=resolution*3600)
    
    # Read the COG
    print(f'Accessing COG ...\n')
    ds = rxr.open_rasterio(f'/vsicurl/{cog_url}')
    
    print(f'Subsetting DEM to bbox ...\n')
    bbox_subset = [
        bbox[0] - bbox_off - resolution,
        bbox[1] - bbox_off - resolution,
        bbox[2] + bbox_off + resolution,
        bbox[3] + bbox_off + resolution
    ]
    dem = ds.rio.clip_box(minx=bbox_subset[0], maxx=bbox_subset[2], miny=bbox_subset[1], maxy=bbox_subset[3])
    # dem = ds.sel(x=slice(bbox_subset[0], bbox_subset[2]), y=slice(bbox_subset[1], bbox_subset[3]))
    
    # Resample to desired resolution
    print(f'Resampling DEM to: {int(resolution*3600)} arc-seconds ...\n')
    
    dem = dem.rio.reproject(
        dem.rio.crs, resolution=resolution, 
        resampling=Resampling.nearest
    )

    dem = dem.to_dataset(name='z')
    nodata_value = dem['z'].rio.nodata
    dem['z'] = dem['z'].where(dem['z'] != nodata_value, np.nan)
    dem = dem.squeeze(dim='band')
    # dem = dem.drop_vars('band')
    print('DEM created successfully!\n')
    
    print(f'Interpolating DEM ...')
    
    bbox_subset = [
        bbox[0] - bbox_off,
        bbox[1] - bbox_off,
        bbox[2] + bbox_off,
        bbox[3] + bbox_off
    ]
    
    minx, maxx = bbox_subset[0], bbox_subset[2]
    miny, maxy = bbox_subset[1], bbox_subset[3]
    
    num_x_points = int((maxx - minx) / (resolution)) + 1
    num_y_points = int((maxy - miny) / (resolution)) + 1
    
    dem = dem.interp(
        x=np.linspace(minx, maxx, num_x_points),
        y=np.linspace(miny, maxy, num_y_points),
        method='nearest'
    )
    
    return dem
        
