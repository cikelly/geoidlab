import requests
import os
import warnings
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
    None
    '''
    if not url:
        url = fetch_url(bbox=bbox)
        
    filename = url.split('/')[-1]
    
    if downloads_dir:
        os.makedirs(downloads_dir, exist_ok=True)
        os.chdir(downloads_dir)
        print(f'Downloading {filename} to {downloads_dir} ...')
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
    
    return 

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

def dem4geoid(bbox):
    '''
    '''