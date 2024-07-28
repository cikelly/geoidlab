import requests
import os

def srtm(bounding_box=None, save_path="srtm_data.tif"):
    base_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",
        "outputFormat": "GTiff"
    }
    
    if bounding_box:
        params.update({
            "south": bounding_box[1],
            "north": bounding_box[3],
            "west": bounding_box[0],
            "east": bounding_box[2]
        })
    
    response = requests.get(base_url, params=params, stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded SRTM data to {save_path}")
    else:
        print("Failed to download data")
        response.raise_for_status()


def srtm_plus():
    '''
    '''
    pass

def cop():
    '''
    '''
    pass

def resample():
    '''
    '''
    pass