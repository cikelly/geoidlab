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

def download_ggm(model_name:str='GO_CONS_GCF_2_TIM_R6e', model_dir='downloads'):
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
    # file_path = os.path.join(model_dir, model_name + '.gfc')
    file_path = model_dir / (model_name + '.gfc')
    
    # Check if file already exists
    if file_path.exists():
    # if os.path.exists(file_path):
        try:
            response = requests.get(base_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.ConnectionError as e:
            print(f"{model_name + '.gfc'} exists, but I cannot verify its completeness due to connectivity issues. Using the existing file.\n")
            return file_path
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching base URL: {e}")
    else:
        try:
            response = requests.get(base_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.ConnectionError as e:
            # print('Please check your internet connection.')
            raise requests.ConnectionError(f'Please check your internet connection. {e}')
    
    model_url = None
    for link in soup.find_all('a', href=True):
        if model_name in link['href'] and 'gfc' in link['href']:
            model_url = model_url_prefix + link['href']
            break

    if not model_url:
        print(f"Model {model_name} not found.")
        return

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching model URL: {e}")
        return
    
    # Ensure output directory exists
    # os.makedirs(model_dir, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    # file_path = os.path.join('downloads', model_name + '.gfc')
    
    # Check if file already exists and has the correct size
    if file_path.exists():
    # if os.path.exists(file_path):
        if file_path.stat().st_size == int(response.headers.get('content-length', 0)):
        # if os.path.getsize(file_path) == int(response.headers.get('content-length', 0)):
            print(f"{model_name + '.gfc'} already exists in \n\t{model_dir} \nand is complete.")
            # print(f"Path: {file_path}")
            return
        else:
            print(f"Model {model_name  + '.gfc'} already exists but is incomplete. Redownloading ...")
            # os.remove(file_path)
            file_path.unlink()
        
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
        except Exception as e:
            print(f"Error during file write: {e}")
            
            # if os.path.exists(file_path):
            #     os.remove(file_path)

    print(f"\n{model_name + '.gfc'} saved to {model_dir}")



def read_icgem(icgem_file:str, model_dir='downloads'):
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
    # Download file if it does not exist
    icgem_file = Path(icgem_file).resolve()
    # if not os.path.exists(icgem_file):
    if not icgem_file.exists():
        # print(f'{icgem_file} cannot be found in {os.getcwd()}. Downloading to {model_dir} ...\n')
        model_name = icgem_file.name.split('.')[0]
        print(f'{model_name+'.gfc'} cannot be found in {Path.cwd()}. Downloading to {model_dir} ...\n')
        download_ggm(model_name)
        # icgem_file = f'{model_dir}' + model_name + '.gfc'
        icgem_file = model_dir / (model_name + '.gfc')
    
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
                values[key] = type_(line.split()[1])
    
    nmax = values.get('max_degree')

    ##### Read Cnm, Snm, sCnm, sSnm
    Cnm = np.zeros( (nmax+1, nmax+1) )
    Snm = np.zeros( (nmax+1, nmax+1) )
    sCnm = np.zeros( (nmax+1, nmax+1) )
    sSnm = np.zeros( (nmax+1, nmax+1) )
    for line in data:
        if line.strip().startswith('gfc'):
            line = line.split()
            n = int(line[1])
            m = int(line[2])
            Cnm[n,m]  = float(line[3])
            Snm[n,m]  = float(line[4]) 
            sCnm[n,m] = float(line[5]) 
            sSnm[n,m] = float(line[6])

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
