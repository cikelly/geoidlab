############################################################
# Utilities for automatic DEM download                     #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import requests
from pathlib import Path
import sys
import warnings
import netCDF4
import re
import certifi

import xarray as xr
import rioxarray as rxr
import numpy as np
from tqdm import tqdm
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

warnings.simplefilter('ignore')

VALID_INTERP_METHODS = {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}

def get_readme_path() -> Path:
    '''
    Function to get the path of README.V11.txt, which is required for automatic DEM download.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    readme_path   : absolute path to the README.V11.txt file
    '''
    script_dir: Path = Path(__file__).resolve().parent
    readme_path: Path = script_dir / '../geoidlab/data/README.V11.txt'
    return readme_path.resolve()

def parse_readme(readme_path) -> list:
    '''
    Parse the README file to extract tile boundary information.

    Parameters
    ----------
    readme_path : Path to the README.V11.txt file.

    Returns
    -------
    tiles       : list of dictionaries containing tile name and boundary info.
    '''
    tiles: list = []
    with open(readme_path, 'r') as file:
        readme_content: str = file.read()
    
    # Regular expression to match the tile information
    tile_pattern: re.Pattern = re.compile(r'(?P<tile>\w+)\s+(?P<min_lat>-?\d+)\s+(?P<max_lat>-?\d+)\s+(?P<min_lon>-?\d+)\s+(?P<max_lon>-?\d+)')
    matches: list = tile_pattern.findall(readme_content)
    
    for match in matches:
        tile_info: dict[str, int] = {
            'tile': match[0],
            'min_lat': int(match[1]),
            'max_lat': int(match[2]),
            'min_lon': int(match[3]),
            'max_lon': int(match[4])
        }
        tiles.append(tile_info)
    
    return tiles

def identify_relevant_tiles(bbox, tiles) -> list:
    '''
    Identify the tiles that intersect with the given bounding box.

    Parameters
    ----------
    bbox           : bbox of the area of interest (to be provided if url is not provided)
    tiles          : list of tiles with their boundaries.

    Returns
    -------
    relevant_tiles : list of tile names that intersect with the bounding box.
    '''
    # Validate bbox
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('bbox must be a list or tuple of 4 values: [min_lon, max_lon, min_lat, max_lat]')
    min_lon, max_lon, min_lat, max_lat = bbox
    
    # Ensure bbox is valid: W, S, E, N
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError('Invalid bbox. Must be [min_lon, min_lat, max_lon, max_lat]')
    
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError('bbox coordinates must be within valid geographic ranges: [-180, 180] for lon, [-90, 90] for lat')
    
    relevant_tiles: list = []

    for tile in tiles:
        if not (tile['max_lat'] < min_lat or tile['min_lat'] > max_lat or
                tile['max_lon'] < min_lon or tile['min_lon'] > max_lon):
            relevant_tiles.append(tile['tile'])
    
    return relevant_tiles

def download_srtm30plus(url=None, downloads_dir=None, bbox=None) -> str:
    '''
    Download SRTM30PLUS from https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/
    
    Parameters
    ----------
    url           : URL of the srtm30plus tile, or a list of URLs if bbox spans multiple tiles.
    downloads_dir : Directory to download the file(s) to.
    bbox          : Bounding box of the area of interest (to be provided if url is not provided).
    
    Returns
    -------
    merged_filepath : str
        Filepath of the merged DEM file if multiple tiles were downloaded.
    '''
    if bbox is None and url is None:
        raise ValueError('Either bbox or url must be provided.')
    
    if not url:
        urls: list[str] = fetch_url(bbox=bbox)
    else:
        urls: list[str] = [url]

    if downloads_dir:
        downloads_dir = Path(downloads_dir)
    else:
        downloads_dir = Path.cwd() / 'downloads'
    downloads_dir = downloads_dir.resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    
    if len(urls) > 1:
        if (downloads_dir / 'merged_dem.nc').resolve().exists():
            if check_bbox_contains(downloads_dir / 'merged_dem.nc', bbox):
                print(f'{downloads_dir}/merged_dem.nc exists and covers bbox. Skip download\n')
                return 'merged_dem.nc'
            else:
                print(f'{downloads_dir}/merged_dem.nc exists but does not cover bbox. Deleting ...\n')
                (downloads_dir / 'merged_dem.nc').unlink(missing_ok=True)
                print(f'Downloading and merging {len(urls)} tiles ...\n')
        
    filepaths: list[str] = []
    for url in urls:
        filename: str = url.split('/')[-1]
        filepath: str = downloads_dir / filename 
        # Check if the file already exists
        if filepath.exists():
            try:
            # 1. If file exists and is readable, use it
                _ = netCDF4.Dataset(filepath)
                if check_bbox_contains(filepath, bbox):
                    print(f'{filename} exists, is readable, and covers bbox. Using local copy.\n')
                    filepaths.append(filepath)
                    continue  # Skip download
                else:
                    print(f'{filename} exists, is readable, but does NOT cover bbox. Redownloading ...\n')
                    filepath.unlink(missing_ok=True)
            except Exception:
                print(f'{filename} exists but is unreadable. Redownloading ...\n')
                filepath.unlink(missing_ok=True)
                
        print(f'Downloading {filename} to: \n\t{downloads_dir} ...')
        # Download NetCDF file
        try:
            response: requests.Response = requests.get(url, verify=certifi.where(), stream=True, timeout=(30, 60))
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                dynamic_ncols=True
            ) as pbar:
                with open(filepath, 'wb') as f:
                    for data in response.iter_content(chunk_size=1024):
                        size: int = f.write(data)
                        pbar.update(len(data))
                        pbar.refresh()
                        f.flush()
                        sys.stdout.flush()
            print('\n')
            # Try to open the file after the download to verify. Skip if unreadable
            try:
                _ = netCDF4.Dataset(filepath)
                filepaths.append(filepath)
            except Exception:
                print(f'Skipping unreadable file {filename}.')
                
        except Exception as e:
            print(f'Download failed for {filename}: {e}.')
            # If file exists but is unreadable, keep it
            if not filepath.exists():
                continue
            else:
                print(f'File {filename} exists but is unreadable and could not be replaced due to download failure.')
                continue

    # Check if any files were downloaded
    if not filepaths:
        raise RuntimeError(
            'No DEM files were downloaded. '
            'Check your internet connection or the availability of the remote server.'
        )
    
    # If multiple files were downloaded, merge them
    if len(filepaths) > 1:
        print('Merging downloaded tiles...')
        datasets: list[xr.Dataset] = [xr.open_dataset(fp) for fp in filepaths]
        # Resolve attribute conflicts
        merged_dataset = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')
        merged_filepath = downloads_dir / 'merged_dem.nc'
        merged_dataset.to_netcdf(merged_filepath)
        return merged_filepath.parts[-1]
    else:
        return filepaths[0].parts[-1]

_tiles_cache = None
def fetch_url(bbox) -> list[str]:
    '''
    Fetch the URLs of all relevant tiles based on the given bounding box.

    Parameters
    ----------
    bbox      : bounding box of the area to download
                    [min_lon, min_lat, max_lon, max_lat]
                    [left, bottom, right, top]

    Returns
    -------
    urls      : url of the srtm30plus tiles
    '''
    # Validate bbox
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('bbox must be a list or tuple of length 4')
    min_lon, max_lon, min_lat, max_lat = bbox

    # Ensure bbox is valid: W, N, E, S
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError('Invalid bbox. Must be [min_lon, min_lat, max_lon, max_lat]')
    
    # Read the README file
    global _tiles_cache
    if _tiles_cache is None:
        readme_path: Path = get_readme_path() 
        _tiles_cache = parse_readme(readme_path)
    tiles: list = parse_readme(readme_path)

    # Identify relevant tiles for the given bbox
    relevant_tiles: list = identify_relevant_tiles(bbox, tiles) 
    
    if not relevant_tiles:
        raise ValueError(f'No SRTM30PLUS tiles found for bbox: {bbox}')

    # Construct the URLs
    base_url = 'https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/'
    urls: list[str] = [f'{base_url}{tile}.nc' for tile in relevant_tiles]

    return urls

def dem4geoid(
    bbox, 
    ncfile=None, 
    bbox_off=1, 
    downloads_dir=None,
    resolution=30,
    model='srtm30plus',
    interp_method='slinear'
) -> xr.Dataset:
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
    model         : name of the DEM model
                     - srtm30plus
                     - srtm
                     - cop
                     - nasadem
                     - gebco   
    interp_method : interpolation method
     
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
    VALID_MODELS = {'srtm30plus', 'srtm', 'cop', 'nasadem', 'gebco'}
    if not bbox:
        raise ValueError('Bounding box must be provided')
    
    if model not in VALID_MODELS:
        raise ValueError(f'Invalid DEM model: {model}. Must be one of:\n{VALID_MODELS}.')
    
    if resolution <=0:
        raise ValueError('Resolution must be a positive integer.')
    
    if interp_method not in VALID_INTERP_METHODS:
        raise ValueError(f'Invalid interp_method: {interp_method}. Must be one of: {VALID_INTERP_METHODS}')
    
    # dictionary of DEM models and function calls
    params = {
        'bbox'         : bbox,
        'model'        : model,
        'downloads_dir': downloads_dir,
        'bbox_off'     : bbox_off,
        'resolution'   : resolution,
        'interp_method': interp_method
    }
    
    models_dict = {
        'srtm30plus': lambda: download_srtm30plus(bbox=bbox, downloads_dir=downloads_dir),
        'srtm'      : lambda: download_dem_cog(**params),
        'cop'       : lambda: download_dem_cog(**params),
        'nasadem'   : lambda: download_dem_cog(**params),
        'gebco'     : lambda: download_dem_cog(**params)
    }

    if downloads_dir:
        downloads_dir = Path(downloads_dir)
    else:
        downloads_dir = Path.cwd() / 'downloads'
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    if not ncfile:
        # Check if a DEM file already exists
        if model == 'srtm30plus':
            urls = fetch_url(bbox)
            if len(urls) > 1:
                expected_filename = 'merged_dem.nc'
            else:
                expected_filename = urls[0].split('/')[-1]
        else:
            expected_filename = f'{model}_dem.nc'
        expected_filepath = downloads_dir / expected_filename
        
        if expected_filepath.exists():
            try:
                if check_bbox_contains(expected_filepath, bbox):
                    print(f'{expected_filename} exists and covers bbox. Using local copy.\n')
                    ncfile = expected_filename
                else:
                    print(f'{expected_filename} exists but does not cover bbox. Redownloading ...\n')
                    expected_filepath.unlink(missing_ok=True)
            except Exception as e:
                print(f'{expected_filename} exists but is unreadable: {e}. Redownloading ...\n')
                expected_filepath.unlink(missing_ok=True)
        
    if not ncfile:
        try:
            if model == 'srtm30plus':
                ncfile = models_dict[model]()
            else:
                dem = models_dict[model]()
                ncfile = expected_filename
                dem.to_netcdf(downloads_dir / ncfile)
                return dem
        except Exception as e:
            raise Exception(
                f'Failed to download {model}: {e}',
                'Specify another model'
            )

    filepath = downloads_dir / ncfile
    
    print(f'Creating xarray dataset of DEM with buffer of {bbox_off} degree(s)\n')    
    
    nc = netCDF4.Dataset(filepath)
    fill_value = nc.variables['z']._FillValue
    
    ds = xr.open_dataset(filepath)
    ds['z']  = ds['z'].where(ds['z'] != fill_value, np.nan)
    
    bbox_subset = [
        bbox[0] - bbox_off,
        bbox[1] + bbox_off,
        bbox[2] - bbox_off,
        bbox[3] + bbox_off
    ]
    # dem = ds.sel(
    #     x=slice(bbox_subset[0], bbox_subset[2]),
    #     y=slice(bbox_subset[1], bbox_subset[3])
    # )
    print(f'Resampling DEM to {resolution} arc-seconds...') if resolution != 30 else None
    minx, maxx = bbox_subset[0], bbox_subset[2]
    miny, maxy = bbox_subset[1], bbox_subset[3]
    
    num_x_points = int((maxx - minx) / (resolution/3600)) + 1
    num_y_points = int((maxy - miny) / (resolution/3600)) + 1
    
    dem = ds.interp(
        x=np.linspace(minx, maxx, num_x_points),
        y=np.linspace(miny, maxy, num_y_points),
        method=interp_method
    )
    
    if dem.rio.crs is None:
        dem.rio.write_crs('EPSG:4326', inplace=True)

    return dem

def download_dem_cog(
    bbox, 
    model='None', 
    cog_url=None,
    bbox_off=2, 
    resolution=30,
    downloads_dir=None,
    interp_method='slinear'
) -> xr.Dataset:
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
    interp_method : interpolation method for resampling
    
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
    model_urls = {
        'srtm'   : 'https://opentopography.s3.sdsc.edu/raster/SRTM_GL3/SRTM_GL3_srtm.vrt',
        'cop'    : 'https://opentopography.s3.sdsc.edu/raster/COP90/COP90_hh.vrt',
        'nasadem': 'https://opentopography.s3.sdsc.edu/raster/NASADEM/NASADEM_be.vrt',
        'gebco'  : 'https://opentopography.s3.sdsc.edu/raster/GEBCOIceTopo/GEBCOIceTopo.vrt'
    }
    if interp_method not in VALID_INTERP_METHODS:
        raise ValueError(f'Invalid interp_method: {interp_method}. Must be one of: {VALID_INTERP_METHODS}')
    
    # Set up downloads directory
    if downloads_dir:
        downloads_dir = Path(downloads_dir).resolve()
    else:
        downloads_dir = Path.cwd() / 'downloads'
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Check if DEM exists and contains bounding box
    ncfile = downloads_dir / f'{model}_dem.nc'

    if ncfile.exists():
        if check_bbox_contains(ncfile, bbox):
            print(f'{ncfile} already exists and contains bounding box. Skipping download')
            return xr.open_dataset(ncfile)
    
    print(f'Downloading {model} DEM to {downloads_dir} ...')  
    
    resolution = resolution / 3600 # convert seconds to degrees
    
    if model is not None and cog_url is None:
        try:
            cog_url = model_urls[model.lower()]
        except KeyError:
            print('Supported models are:\nsrtm\ncop\nnasadem\ngebco.')
            raise ValueError(f'Unsupported model: {model}')
    
    # Read the COG
    print(f'Accessing {model.upper()} COG at {model_urls[model.lower()]}\n')
    
    for attempt in range(3):
        try:
            ds = rxr.open_rasterio(f'/vsicurl/{cog_url}')
            break
        except RasterioIOError as e:
            print(f'Attempt {attempt+1} failed. Error: {e}.')
            if attempt < 2:
                import time
                print(f'Retrying in 5 seconds...')
                time.sleep(5)
            else:
                raise RuntimeError(f'Failed to access COG for {model}: {e}.')

    bbox_subset = [
        bbox[0] - bbox_off,
        bbox[1] + bbox_off,
        bbox[2] - bbox_off,
        bbox[3] + bbox_off
    ]
    dem = ds.rio.clip_box(minx=bbox_subset[0], maxx=bbox_subset[2], miny=bbox_subset[1], maxy=bbox_subset[3])
    
    if not isinstance(dem, xr.Dataset):
        dem = dem.to_dataset(name='z')
    
    # Intercept the nodata value before accessing it
    try:
        nodata_value = dem['z'].rio.nodata
    except OverflowError:
        nodata_value = np.finfo(np.float32).max
        dem['z'].rio.write_nodata(nodata_value, inplace=True)
    
    # Convert nodata value to a manageable value
    if nodata_value is None or (isinstance(nodata_value, (int, float)) and nodata_value > np.finfo(np.float32).max):
        nodata_value = np.finfo(np.float32).max
        dem['z'].rio.write_nodata(nodata_value, inplace=True)
        
    dem['z'] = dem['z'].where(dem['z'] != nodata_value, np.nan)
    dem = dem.squeeze(dim='band')
    # dem = dem.drop_vars('band')
    print('DEM created successfully!\n')
    
    print(f'Interpolating DEM ...')
    
    minx, maxx = bbox_subset[0], bbox_subset[2]
    miny, maxy = bbox_subset[1], bbox_subset[3]
    
    num_x_points = int((maxx - minx) / (resolution)) + 1
    num_y_points = int((maxy - miny) / (resolution)) + 1
    
    dem = dem.interp(
        x=np.linspace(minx, maxx, num_x_points),
        y=np.linspace(miny, maxy, num_y_points),
        method=interp_method
    )
    
    dem.to_netcdf(ncfile)

    return dem



def check_bbox_contains(netcdf_file, bbox) -> bool:
    '''
    Check if a NetCDF file contains a bounding box
    
    Parameters
    ----------
    netcdf_file : path to the NetCDF file
    bbox        : bounding box of the area of interest (W, S, E, N)
    
    Returns
    -------
    contains    : bool
    '''
    # Load the NetCDF file
    try:
        ds = xr.open_dataset(netcdf_file)
    except Exception as e:
        raise RuntimeError(f'Failed to open {netcdf_file}: {e}')
    
    if 'x' not in ds.coords or 'y' not in ds.coords:
        raise ValueError(f'NetCDF file {netcdf_file} does not contain x and y coordinates')
    
    # Extract the bounding box coordinates from the NetCDF file
    x_min = ds['x'].min().item()
    x_max = ds['x'].max().item()
    y_min = ds['y'].min().item()
    y_max = ds['y'].max().item()
    
    # Given bounding box coordinates (WSEN)
    bbox_w, bbox_e, bbox_s, bbox_n = bbox
    
    # Check for intersection
    # intersects = not (bbox_e < x_min or bbox_w > x_max or bbox_n < y_min or bbox_s > y_max)
    
    # Check for containment
    contains = (bbox_w >= x_min and bbox_e <= x_max and bbox_s >= y_min and bbox_n <= y_max)
    
    return contains



# def download_srtm30plus(url=None, downloads_dir=None, bbox=None):
#     '''
#     Download SRTM30PLUS from https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/
    
#     Parameters
#     ----------
#     url           : url of the srtm30plus tile
#     downloads_dir : directory to download the file to
#     bbox          : bbox of the area of interest (to be provided if url is not provided)
    
#     Returns
#     -------
#     fname         : Name of downloaded file
#     '''
#     if not url:
#         url = fetch_url(bbox=bbox)
        
#     filename = url.split('/')[-1]
#     filepath = os.path.join(downloads_dir, filename) if downloads_dir else filename
    
#     # Check if the file already exists
#     if os.path.exists(filepath):
#         try:
#             response_head = requests.head(url, verify=False)
#             total_size = int(response_head.headers.get('content-length', 0))
#             # Check if the existing file size matches the expected size
#             if os.path.getsize(filepath) == total_size:
#                 print(f'{filename} already exists and is complete. Skip download\n')
#                 return filename
#             else:
#                 print(f'{filename} already exists but is incomplete. Redownloading ...\n')
#                 os.remove(filepath)
#         except requests.exceptions.RequestException:
#             print(f'Unable to check if {filename} is complete. {filename} in {downloads_dir} will be used ...\n')
#             return filename
            
#     if downloads_dir:
#         os.makedirs(downloads_dir, exist_ok=True)
#         print(f'Downloading {filename} to {downloads_dir} ...\n')
#     else:
#         print(f'Downloading {filename} to {os.getcwd()} ...\n')
    
#     # Download NetCDF file
#     try:
#         response = requests.get(url, verify=False, stream=True)
#         total_size = int(response.headers.get('content-length', 0))
        
#         with tqdm(
#             # desc=filename,
#             total=total_size,
#             unit='iB',
#             unit_scale=True,
#             dynamic_ncols=True
#         ) as pbar:
            
#             with open(filepath, 'wb') as f:
#                 for data in response.iter_content(chunk_size=1024):
#                     size = f.write(data)
#                     pbar.update(len(data))
#                     pbar.refresh()
#                     f.flush()
#                     sys.stdout.flush()
#     except Exception as e:
#             raise RuntimeError(f'Download failed: {e}. Are you connected to the internet?')
                    
#     return filename

# def fetch_url(bbox):
#     '''
#     Get the url of the srtm30plus tiles
    
#     Parameters
#     ----------
#     bbox          : bounding box of the area to download
#                         [min_lon, min_lat, max_lon, max_lat]
#                         [left, bottom, right, top]
#     downloads_dir : directory to download the file to
    
#     Returns
#     -------
#     url           : url of the srtm30plus tile
#     '''
#     readme_path = get_readme_path()
#     # Define the base URL
#     base_url = 'https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/'

#     # Read the tile boundaries from the README file
#     tiles = {}
#     with open(readme_path, 'r') as f:
#         for line in f:
#             parts = line.split()
#             if len(parts) == 5 and parts[0][0] in 'we':
#                 tile = parts[0]
#                 lat_min, lat_max = map(int, parts[1:3])
#                 lon_min, lon_max = map(int, parts[3:5])
#                 tiles[tile] = {'lon': (lon_min, lon_max), 'lat': (lat_min, lat_max)}

#     # Find the tile that contains the bounding box
#     for tile, bounds in tiles.items():
#         if (bounds['lon'][0] <= bbox[0] < bounds['lon'][1] and
#             bounds['lat'][0] <= bbox[1] < bounds['lat'][1]):
#             return base_url + tile + '.nc'

#     raise ValueError('No tile found that contains the bounding box')