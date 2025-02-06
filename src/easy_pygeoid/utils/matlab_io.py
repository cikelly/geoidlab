############################################################
# Utilities for reading and writing mat files              #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np
import xarray as xr
import h5py

from scipy.io import loadmat, savemat

class MATIO():
    '''
    Class to read and write mat files
    '''
    def __init__(self, filename) -> None:
        '''
        Initialize
        
        Parameters
        ----------
        filename  : (str) path to the mat file
        
        Returns
        -------
        None
        '''
        self.filename = filename
    def read_mat_v5(self) -> None:
        '''
        Read version 5 mat file
        
        Returns
        -------
        None
        '''
        data = loadmat(self.filename)

        keys = [key for key in data.keys() if not key.startswith('__')]

        for key in keys:
            if 'lat' in key.lower():
                lat = data[key]
            elif 'lon' in key.lower():
                lon = data[key]

        if lon.ndim == 2:
            if lon.shape[0] == 1 or lon.shape[1] == 1:
                lon = lon.flatten()
            if lat.shape[0] == 1 or lat.shape[1] == 1:
                lat = lat.flatten()
            else:
                lat = lat[:, 0]
                lon = lon[0, :]
            
        data_vars = {}
        for key in keys:
            if data[key].shape == (len(lat), len(lon)) and ('lat' not in key.lower() and 'lon' not in key.lower()):
                data_vars[key.lower()] = (['lat', 'lon'], data[key])

        self.lon = lon
        self.lat = lat
        self.data_vars = data_vars
    
    def read_mat_v7(self) -> None:
        '''
        '''
        data = h5py.File(self.filename)
        keys = [key for key in data.keys() if not key.startswith('__')]

        for key in keys:
            if 'lat' in key.lower():
                lat = data[key][:]
            elif 'lon' in key.lower():
                lon = data[key][:]    

        if lon.ndim == 2:
            if lon.shape[0] == 1 or lon.shape[1] == 1:
                lon = lon.flatten()
            if lat.shape[0] == 1 or lat.shape[1] == 1:
                lat = lat.flatten()
            else:
                lat = lat[:, 0]
                lon = lon[0, :]
            
        data_vars = {}
        for key in keys:
            if data[key].shape == (len(lat), len(lon)) and ('lat' not in key.lower() and 'lon' not in key.lower()):
                data_vars[key.lower()] = (['lat', 'lon'], data[key][:])

        self.lon = lon
        self.lat = lat
        self.data_vars = data_vars
        
    def _to_xarray(self) -> xr.Dataset:
        '''
        Convert to xarray dataset object
        
        Returns
        -------
        xr_dataset
        '''
        xr_dataset = xr.Dataset(
            self.data_vars,
            coords={
                'lat': self.lat,
                'lon': self.lon,
            }
        )
        
        return xr_dataset
    
    def read_mat(self, to_xarray: bool=True) -> xr.Dataset | None:
        '''
        Read mat file
        
        Returns
        -------
        xr_dataset
        '''
        try:
            self.read_mat_v7()
        except:
            self.read_mat_v5()
        
        return self._to_xarray() if to_xarray else None
        