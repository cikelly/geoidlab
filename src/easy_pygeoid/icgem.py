############################################################
# Utilities downloading and reading ICGEM gfc format       #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
# import os
from pathlib import Path
import numpy as np

def download_ggm(model_name: str = 'GO_CONS_GCF_2_TIM_R6e', model_dir='downloads') -> None:
    '''
    Download static gravity model from ICGEM
    
    Parameters
    ----------
    model_name: (str) Name of global model
    model_dir : (str) Directory to download model to
    
    Returns
    -------
    None
    
    Notes
    -----
    1. Automatically writes global model to file
    '''
    base_url = "https://icgem.gfz-potsdam.de/tom_longtime"
    model_url_prefix = 'https://icgem.gfz-potsdam.de'
    model_dir = Path(model_dir).resolve()
    file_path = model_dir / (model_name + '.gfc')
    
    def validate_gfc_file(file_path) -> bool:
        '''Check if the file is a valid .gfc file.'''
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Check for expected .gfc keywords
                return any('product_type' in line or 'earth_gravity_constant' in line for line in lines)
        except (IOError, UnicodeDecodeError):
            return False
    
    # Check if file already exists and is valid
    if file_path.exists():
        if validate_gfc_file(file_path):
            try:
                # Try to connect to check for updates
                response = requests.get(base_url, timeout=10)
                response.raise_for_status()
                print(f'{model_name}.gfc exists in \n\t{model_dir} \nand is valid.\n')
                return 
            except requests.RequestException:
                # If can't connect but file is valid, use existing file
                print(f'{model_name}.gfc exists and appears valid. Using the existing file.\n')
                return 
        else:
            # File exists but is invalid
            print(f'{model_name}.gfc exists but is invalid or corrupted. Redownloading...\n')
            file_path.unlink()
    
    # Connect to server to get model listing
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        raise requests.RequestException(f'Please check your internet connection. {e}')
    
    # Find the model URL
    model_url = None
    for link in soup.find_all('a', href=True):
        if model_name in link['href'] and 'gfc' in link['href']:
            model_url = model_url_prefix + link['href']
            break

    if not model_url:
        raise ValueError(f'Model {model_name} not found on ICGEM server.')

    # Get the model file
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        raise requests.RequestException(f'Error fetching model URL: {e}')
    
    # Ensure output directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with tqdm(total=total_size, unit='iB', desc=model_name, unit_scale=True) as pbar:
        try:
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)
            if total_size != 0 and pbar.n != total_size:
                print("ERROR, something went wrong during the download.")
                if file_path.exists():
                    file_path.unlink()
                raise IOError(f"Failed to download {model_name}.gfc")
        except (KeyboardInterrupt, Exception) as e:
            print(f'\nDownload interrupted or failed: {e}')
            if file_path.exists():
                file_path.unlink()  # Remove partial file
            raise IOError(f"Failed to download {model_name}.gfc")
            
    # Verify the downloaded file
    if not validate_gfc_file(file_path):
        print(f'Downloaded file {model_name}.gfc is invalid.')
        file_path.unlink()
        raise IOError(f"Downloaded file {model_name}.gfc is invalid.")

    print(f"\n{model_name}.gfc saved to {model_dir}")
    return 



def read_icgem(icgem_file:str, model_dir='downloads') -> dict:
    '''
    Read spherical harmonic coefficients from an ICGEM .gfc file.

    Parameters
    ----------
    icgem_file : str
        The path to the ICGEM .gfc file.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'a'       : The reference radius.
        - 'nmax'    : The maximum degree of expansion.
        - 'GM'      : The Earth's gravitational constant.
        - 'Cnm'     : A numpy array containing the cosine coefficients.
        - 'Snm'     : A numpy array containing the sine coefficients.
        - 'sCnm'    : A numpy array containing the formal cosine errors.
        - 'sSnm'    : A numpy array containing the formal sine errors.
        - 'tide_sys': The permanent tide system of the model.
    '''
    
    def fortran_to_float(fortran_str) -> float:
        '''Replace 'd' or 'D' (Fortran double precision exponent) with 'e' (Python float exponent)'''
        return float(fortran_str.lower().replace('d', 'e'))

    # Download file if it does not exist
    icgem_file = Path(icgem_file).resolve()
    # if not os.path.exists(icgem_file):
    if not icgem_file.exists():
        model_name = icgem_file.stem
        # print(f'{model_name+'.gfc'} cannot be found in {Path.cwd()}. Downloading to {model_dir} ...\n')
        download_ggm(model_name)
        # icgem_file = f'{model_dir}' + model_name + '.gfc'
        icgem_file = Path(model_dir) / (model_name + '.gfc')
    
    with open(icgem_file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()
    
    ##### Read a, GM, nmax
    keys = {
        'earth_gravity_constant': float,
        'radius': float,
        'max_degree': int,
        'tide_system': str
    }

    values = {}
    for line in data:
        for key, type_ in keys.items():
            if key in line:
                values[key] = line.split()[1]
                
                if type_ == float:
                    try:
                        values[key] = fortran_to_float(values[key])
                    except ValueError:
                        values[key] = type_(values[key])
                else:
                    values[key] = type_(values[key])
    
    nmax = values.get('max_degree')

    ##### Read Cnm, Snm, sCnm, sSnm
    Cnm  = np.zeros((nmax+1, nmax+1))
    Snm  = np.zeros((nmax+1, nmax+1))
    sCnm = np.zeros((nmax+1, nmax+1))
    sSnm = np.zeros((nmax+1, nmax+1))
    
    for line in data:
        if line.strip().startswith('gfc'):
            parts = line.split()
            if len(parts) >= 7:
                n = int(parts[1])
                m = int(parts[2])
                try:
                    Cnm[n,m]  = fortran_to_float(parts[3])
                    Snm[n,m]  = fortran_to_float(parts[4]) 
                    sCnm[n,m] = fortran_to_float(parts[5]) 
                    sSnm[n,m] = fortran_to_float(parts[6])
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line.strip()}")
                    print(f"Error message: {e}")

    shc             = {}
    shc['a']        = values.get('radius')
    shc['nmax']     = nmax
    shc['GM']       = values.get('earth_gravity_constant')
    shc['Cnm']      = Cnm
    shc['Snm']      = Snm
    shc['sCnm']     = sCnm
    shc['sSnm']     = sSnm
    shc['tide_sys'] = values.get('tide_system')

    return shc
