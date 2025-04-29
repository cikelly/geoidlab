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
    def __init__(
        self, 
        filename:str = None, 
        xr_data: xr.Dataset = None, 
        save_filename:str = None,
        **kwargs
    ) -> None:
        '''
        Initialize
        
        Parameters
        ----------
        filename      : (str) path to the mat file
        xr_data       : (xr.Dataset) xarray dataset
        save_filename : (str) path to save the mat file
        **kwargs      : additional keyword arguments for use in write_xarray
        
        Returns
        -------
        None
        '''
        self.data = None
        self.xrdata = xr_data
        self.filename = filename
        self.save_filename = save_filename
        self.kwargs = kwargs
        
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
    
    def read_mat(self, to_xarray=True) -> xr.Dataset | None:
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
            return self.write_xarray()
        else:
            return None

    def write_xarray(self, anomaly_var=None, anomaly=False) -> xr.Dataset:
        '''
        Convert to xarray dataset object
        
        Parameters
        ----------
        anomaly_var : name of the anomaly variable (fall back to **kwargs in __init__)
        anomaly     : flag for whether data is gravity anomalies (fall back to **kwargs in __init__)
        
        Returns
        -------
        xr_dataset
        
        Notes
        -----
        1. Designed to output data ready for Stokes' formula if gravity anomaly data is provided
        '''
        if self.data_vars is None:
            raise ValueError('No data variables available. Run read_mat first.')
        
        # Extract anomaly parameters
        anomaly_var = anomaly_var if anomaly_var is not None else self.kwargs.get('anomaly_var', None)
        anomaly     = anomaly if anomaly else self.kwargs.get('anomaly', False)
        
        # Handle anomaly variable renaming
        if anomaly and anomaly_var is not None:
            if anomaly_var in self.data_vars:
                self.data_vars['Dg'] = self.data_vars.pop(anomaly_var)
            elif 'dg' in self.data_vars:
                self.data_vars['Dg'] = self.data_vars.pop('dg')
        elif 'dg' in self.data_vars:
            self.data_vars['Dg'] = self.data_vars.pop('dg')
            
        try:
            xr_dataset = xr.Dataset(
                self.data_vars,
                coords={
                    'lat': (['lat'], self.lat),
                    'lon': (['lon'], self.lon),
                }
            )
        except Exception as e:
            raise ValueError(f'Failed to create xarray Dataset: {str(e)}')
        
        return xr_dataset
    
    def write_mat(self) -> None:
        '''
        Write a MAT file from the xarray Dataset object
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Notes
        -----
        1. Writes a MAT file with the same structure as the input xarray Dataset object
        '''
        # Create meshgrid of longitude and latitude arrays
        Lon, Lat = np.meshgrid(self.xrdata.lon.values, self.xrdata.lat.values)
        
        # Initialize data dictionary
        data_vars = {'Long': Lon, 'Lat': Lat}
        
        # Iterate through data variables and add to dictionary
        for var in self.xrdata.data_vars:
            data_vars[var] = self.xrdata[var].values
        
        # Write data dictionary to MAT file
        savemat(self.save_filename, data_vars)
        
