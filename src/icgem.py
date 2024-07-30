############################################################
# Constants for gravity field modelling                    #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

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
    
    # Ensure output directory exists
    os.makedirs('downloads', exist_ok=True)
    file_path = os.path.join('downloads', model_name + '.gfc')
    
    # Check if file already exists and has the correct size
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == int(response.headers.get('content-length', 0)):
            print(f"Model {model_name} already downloaded.")
            print(f"Path: {file_path}")
            return
        else:
            print(f"Model {model_name} already exists but is incomplete. Redownloading ...")
            os.remove(file_path)
        
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
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

    print(f"Downloaded model saved to {file_path}")
