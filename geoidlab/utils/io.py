############################################################
# Utilities for reading and writing                        #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime

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

def save_to_netcdf(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dataset_key: str,
    proj_dir: str = None, 
    overwrite: bool = True,
    filepath: str = None
) -> None:
    '''
    Save a dataset to a NetCDF file using predefined or default configuration
    
    Parameters
    ----------
    data      : the data to save
    lon       : longitude 
    lat       : latitude
    proj_dir  : Directory to save data to
    filepath  : If filepath is provided, prefer it over proj_dir
    overwrite : Overwrite existing file if it exists
    
    Returns
    -------
    None
    '''
    # Select configuration
    config = DATASET_CONFIG.get(dataset_key, DEFAULT_CONFIG)
    
    try:
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
                'units': config['units'],
                'description': config['description'],
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'created_by': 'geoidlab',
                'website': 'https://github.com/cikelly/geoidlab',
                'copyright': f'Copyright (c) {datetime.now().year}, Caleb Kelly',
            }
        )
        
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




