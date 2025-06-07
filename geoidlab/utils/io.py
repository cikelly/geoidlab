############################################################
# Utilities for reading and writing                        #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
from tzlocal import get_localzone

DATASET_CONFIG = {
    'tc': {
        'var_name'   : 'tc',
        'units'      : 'mGal',
        'description': 'Terrain Correction',
        'fname'      : 'TC',
        'long_name'  : 'Terrain Correction',
    },
    'N_ind': {
        'var_name'   : 'N_ind',
        'units'      : 'meters',
        'description': 'Indirect Effect of Helmert\'s condensation on the geoid',
        'fname'      : 'N_ind',
        'long_name'  : 'Indirect Effect',
    },
    'rtm': {
        'var_name'   : 'rtm_anomaly',
        'units'      : 'mGal',
        'description': 'Residual Terrain Model (RTM) Gravity Anomalies',
        'fname'      : 'RTM',
        'long_name'  : 'Residual Terrain Model Gravity Anomalies',
    },
    'zeta': {
        'var_name'   : 'zeta',
        'units'      : 'meters',
        'description': 'Height anomaly estimated from a global geopotential model',
        'fname'      : 'zeta',
        'long_name'  : 'Height anomaly',
    },
    'Dg_ggm': {
        'var_name'   : 'Dg',
        'units'      : 'mGal',
        'description': 'Gravity anomaly synthesized from a global geopotential model (GGM)',
        'fname'      : 'Dg_ggm',
        'long_name'  : 'Gravity anomaly',
    },
    'N_ref': {
        'var_name'   : 'N_ref',
        'units'      : 'm',
        'description': 'Geoid height synthesized from a global geopotential model (GGM)',
        'fname'      : 'N_ref',
        'long_name'  : 'Geoid Height',
    },
    'dg': {
        'var_name'   : 'dg',
        'units'      : 'mGal',
        'description': 'Gravity disturbance synthesized from a global geopotential model (GGM)',
        'fname'      : 'dg',
        'long_name'  : 'Gravity Disturbance',
    },
    'zeta_rtm':{
        'var_name'   : 'zeta_rtm',
        'units'      : 'm',
        'description': 'RTM height anomaly',
        'fname'      : 'zeta_rtm',
        'long_name'  : 'RTM Height Anomaly'
    },
    'N_res': {
        'var_name'   : 'N_res',
        'units'      : 'm',
        'description': 'Residual geoid height',
        'fname'      : 'N_res',
        'long_name'  : 'Residual geoid height'
    },
    'N': {
        'var_name'   : 'N',
        'units'      : 'm',
        'description': 'Geoid height computed as the sum of the residual geoid, the reference geoid, and the indirect effect',
        'fname'      : 'N',
        'long_name'  : 'Geoid Height'
    },
    'free_air': {
        'var_name'   : 'free_air',
        'units'      : 'mGal',
        'description': 'Free-air gravity anomaly',
        'fname'      : 'free_air_gridded',
        'long_name'  : 'Free-air Gravity Anomaly',
    },
    'bouguer': {
        'var_name'   : 'bouguer',
        'units'      : 'mGal',
        'description': 'Bouguer gravity anomaly',
        'fname'      : 'bouguer_gridded',
        'long_name'  : 'Bouguer Gravity Anomaly',
    },
    'Dg': {
        'var_name'   : 'Dg',
        'units'      : 'mGal',
        'description': 'Helmert gravity anomaly',
        'fname'      : 'gridded_anomalies',
        'long_name'  : 'Helmert Gravity Anomaly',
    },
    'Dg_SITE': {
        'var_name'   : 'Dg_SITE',
        'units'      : 'mGal',
        'description': 'Secondary indirect effect of Helmert\'s condensation on gravity',
        'fname'      : 'Dg_SITE',
        'long_name'  : 'Secondary Indirect Effect on Gravity',
    },
    'Dg_ELL': {
        'var_name'   : 'Dg_ELL',
        'units'      : 'mGal',
        'description': 'Ellipsoidal correction for gravity anomalies',
        'fname'      : 'Dg_ELL',
        'long_name'  : 'Ellipsoidal Correction on Gravity',
    }
    # Add more datasets as needed
}

# Generic fallback configuration
DEFAULT_CONFIG = {
    'var_name'       : 'data',
    'units'          : 'unknown',
    'description'    : 'Generic Dataset',
    'fname'          : 'Generic',
    'long_name'      : 'Generic Dataset',
}

TIDE_SYSTEM_DATASETS = {'Dg_ggm', 'N_ref', 'dg', 'N_res', 'N', 'zeta', 'free_air', 'bouguer', 'Dg'}

METHOD_DATASETS = {'N', 'N_res', 'N_ref'}

def save_to_netcdf(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dataset_key: str,
    proj_dir: str = None, 
    overwrite: bool = True,
    filepath: str = None,
    tide_system: str = None,
    method: str = None
) -> str:
    '''
    Save a dataset to a NetCDF file using predefined or default configuration
    
    Parameters
    ----------
    data        : the data to save
    lon         : longitude 
    lat         : latitude
    proj_dir    : Directory to save data to
    filepath    : If filepath is provided, prefer it over proj_dir
    overwrite   : Overwrite existing file if it exists
    tide_system : Tide system of the data (only for specific datasets)
    method      : Method used to compute the data (only for specific datasets)
    dataset_key : Key to select the dataset configuration from DATASET_CONFIG
    
    Returns
    -------
    str         : 'Success' or 'Failed'
    '''
    # Select configuration
    config = DATASET_CONFIG.get(dataset_key, DEFAULT_CONFIG)
    
    try:
        local_tz = get_localzone()
        date_created = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
        # Set up working directory (optional)
        if filepath is None:
            if proj_dir is None:
                proj_dir = Path.cwd()
            else:
                proj_dir = Path(proj_dir)
            
            # Set up save directory and filename
            save_dir = proj_dir / 'results'
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / Path(config['fname'] + '.nc')
        else:
            filename = Path(filepath)
        
        # Ensure lon and lat are 1D arrays
        if lon.ndim == 2:
            lon = lon[0, :]
            lat = lat[:, 0]
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                config['var_name']: (
                    ['lat', 'lon'], 
                    data, 
                    {
                        'units': config['units'],
                        'long_name': config['long_name'],
                    }
                ),
            },
            coords={
                'lat': (['lat'], lat, {'long_name': 'latitude'}),
                'lon': (['lon'], lon, {'long_name': 'longitude'}),
            },
            attrs={
                'units'       : config['units'],
                'description' : config['description'],
                # 'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date_created': f'{date_created} {local_tz()}',
                'created_by'  : 'geoidlab',
                'website'     : 'https://github.com/cikelly/geoidlab',
                'copyright'   : f'Copyright (c) {datetime.now().year}, Caleb Kelly',
            }
        )
        
        # Add tide system information only for relevant datasets
        if dataset_key in TIDE_SYSTEM_DATASETS and tide_system is not None:
            ds.attrs['tide_system'] = tide_system
        
        # Add method
        if dataset_key in METHOD_DATASETS:
            ds.attrs['method'] = method if method is not None else 'Unspecified method'
        
        # Save to NetCDF file
        if filename.exists() and not overwrite:
            print(f'File {filename} already exists. Use overwrite=True to replace it.')
            return

        ds.to_netcdf(filename, mode='w')
        
        return 'Success'
    
    except PermissionError as e:
        print(f'Warning: Permission denined: {filename}. Please close the file and try again.')
        print(f'Error details: {str(e)}')
        # return
    except OSError as e:
        print(f'Warning: Failed to write to {filename}.')
        print(f'Error details: {str(e)}')
        # return
    except Exception as e:
        print(f'Warning: An unexpected error occurred while saving {filename}.')
        print(f'Error details: {str(e)}')
        # return
    
    return 'Failed'
    # ds.to_netcdf(filename, mode='w')




