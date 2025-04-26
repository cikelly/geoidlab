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
    'IND': {
        'var_name'   : 'ind',
        'units'      : 'meters',
        'description': 'Indirect Effect',
        'fname'      : 'IND',
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
        'description': 'Height anomaly due to residual topography',
        'fname'      : 'zeta',
        'long_name'  : 'Height anomaly',
    },
    # Add more datasets as needed
}

# Generic fallback configuration
DEFAULT_CONFIG = {
    'var_name'       : 'data',
    'units'          : 'unknown',
    'description'    : 'Generic Dataset',
    'fname'          : 'Generic'
}

def save_to_netcdf(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dataset_key: str,
    proj_dir: str = None, 
) -> None:
    '''
    Save a dataset to a NetCDF file using predefined or default configuration
    
    Parameters
    ----------
    data      : the data to save
    lon       : longitude 
    lat       : latitude
    proj_dir  : Directory to save data to
    filename  : Name of file to use for the saved data
    
    Returns
    -------
    None
    '''
    # Set up working directory
    if proj_dir is None:
        proj_dir = Path.cwd()
    else:
        proj_dir = Path(proj_dir)
    
    # Select configuration
    config = DATASET_CONFIG.get(dataset_key, DEFAULT_CONFIG)
    
    # Set up save directory and filename
    save_dir = proj_dir / 'results'
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / Path(config['fname'] + '.nc')
    
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
    ds.to_netcdf(filename)




