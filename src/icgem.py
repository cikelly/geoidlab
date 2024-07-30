############################################################
# Constants for gravity field modelling                    #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from numpy import (
    zeros
)

def download_ggm(model_name: str = 'GO_CONS_GCF_2_TIM_R6e'):
    '''
    Download static gravity model from ICGEM
    
    Parameters
    ----------
    model_name: (str) Name of global model
    
    Returns
    -------
    None
    
    Notes
    -----
    1. Automatically writes global model to file
    '''
    base_url = "https://icgem.gfz-potsdam.de/tom_longtime"
    model_url_prefix = 'https://icgem.gfz-potsdam.de'

    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    except requests.RequestException as e:
        print(f"Error fetching base URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
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
    
    print(f'Downloading {model_name + '.gfc'} ...\n')
    # Ensure output directory exists
    os.makedirs('downloads', exist_ok=True)
    file_path = os.path.join('downloads', model_name + '.gfc')
    
    # Check if file already exists and has the correct size
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == int(response.headers.get('content-length', 0)):
            print(f"{model_name + '.gfc'} already exists and is complete.")
            # print(f"Path: {file_path}")
            return
        else:
            print(f"Model {model_name  + '.gfc'} already exists but is incomplete. Redownloading ...")
            os.remove(file_path)
        
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

    print(f"\n{model_name + '.gfc'} saved to downloads")



def read_icgem(icgem_file:str):
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
        - 'tide_sys': The tide system used in the model.
    '''
    with open(icgem_file, 'r') as f:
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
    Cnm = zeros( (nmax+1, nmax+1) )
    Snm = zeros( (nmax+1, nmax+1) )
    sCnm = zeros( (nmax+1, nmax+1) )
    sSnm = zeros( (nmax+1, nmax+1) )
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