############################################################
# Utilities for reading and writing mat files              #
# Copyright (c) 2025, Caleb Kelly                          #
# Author: Caleb Kelly  (2025)                              #
############################################################

import numpy as np
import xarray as xr
import h5py

from scipy.io import loadmat, savemat

class MATLABIO():
    '''
    Class to read and write mat files
    '''
    def __init__(self, filename:str = None, xr_data: xr.Dataset = None, save_filename:str = None) -> None:
        '''
        Initialize
        
        Parameters
        ----------
        filename  : (str) path to the mat file
        
        Returns
        -------
        None
        '''
        self.data = None
        self.xrdata = xr_data
        self.filename = filename
        self.save_filename = save_filename
        
    def read_mat_v5(self) -> None:
        '''
        Read version 5 mat file
        
        Returns
        -------
        None
        '''
        data = loadmat(self.filename)
        keys = [key for key in data.keys() if not key.startswith('__')]
        lat = None
        lon = None

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
        lat = None
        lon = None

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
        
    def read_single_variable(self) -> None:
        '''
        Read a MAT file with a single variable
        
        Returns
        -------
        None
        '''
        def _read_v7() -> None:
            data = h5py.File(self.filename, 'r')
            keys = [key for key in data.keys() if not key.startswith('__')]
            if len(keys) == 1:
                self.data = data[keys[0]][:]
            else:
                raise ValueError('Expected a single variable, but found multiple.')
        
        def _read_v5() -> None:
            data = loadmat(self.filename)
            keys = [key for key in data.keys() if not key.startswith('__')]
            if len(keys) == 1:
                self.data = data[keys[0]]
            else:
                raise ValueError('Expected a single variable, but found multiple.')
        
        try:
            _read_v7()
        except OSError:
            _read_v5()
        
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
            try:
                self.read_mat_v5()
            except:
                try:
                    self.read_single_variable()
                except OSError as e:
                    raise e
        
        if self.data is not None:
            return self.data  # Return single variable directly
        elif to_xarray:
            return self._to_xarray()
        else:
            return None
    
    def write_mat(self) -> None:
        '''
        Write mat file
        
        Returns
        -------
        None
        '''
        Lon, Lat = np.meshgrid(self.xrdata.lon.values, self.xrdata.lat.values)
        
        data_vars = {'Long': Lon, 'Lat': Lat}
        for var in self.xrdata.data_vars:
            data_vars[var] = self.xrdata[var].values
        savemat(self.save_filename, data_vars)