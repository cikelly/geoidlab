############################################################
# Utilities for converting between tide systems            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import pandas as pd
import numpy as np

from pathlib import Path

from geoidlab import constants
from geoidlab.coordinates import geodetic2geocentric

class GravityTideSystemConverter:
    '''
    Convert between different permanet tide systems
    
    Parameters
    ----------
    path_to_data      : path to data file
    data              : numpy array or Pandas DataFrame or dict
                        (lon, lat, elevation, gravity)
    
    Notes
    -----
    1. Arrange your data in the order: lon, lat, elevation, gravity
    2. Supported file formats: .csv, .txt, .xlsx, .xls
    3. If both path_to_data and data are provided, data will be used
    '''
    VALID_FILE_TYPES = ['csv', 'txt', 'xlsx', 'xls']
    
    def __init__(self, path_to_data=None, data=None) -> None:
        '''
        Initialize GravityTideSystemConverter
        
        Parameters
        ----------
        path_to_data      : path to data file
        data              : numpy array or Pandas DataFrame or dict
                            (lon, lat, elevation, gravity)
        '''
        # Input validation
        if path_to_data is None and data is None:
            raise ValueError('Please provide either path to data or data.')
        
        self.path_to_data = Path(path_to_data) if path_to_data is not None else None
        self.data         = data if data is not None else self.read_file()
        
        if self.path_to_data is not None:
            self.data = self.read_file()
        else:
            if self.data is not None:
                if isinstance(self.data, pd.DataFrame):
                    self.data.columns = ['lon', 'lat', 'h', 'gravity']
                elif isinstance(self.data, np.ndarray):
                    self.data = pd.DataFrame(self.data, columns=['lon', 'lat', 'h', 'gravity'])
                elif isinstance(self.data, dict):
                    self.data = pd.DataFrame(self.data)
                    self.data.columns = ['lon', 'lat', 'h', 'gravity']
                else:
                    raise ValueError('Data must be a Pandas DataFrame, numpy array, or dictionary.')
        
        # Precompute terms in bracket
        self.g_bracket = -30.4 + 91.2 * np.sin(np.radians(self.data['lat'])) ** 2
        self.h_bracket = -0.198 * (3/2 * np.sin(np.radians(self.data['lat']))**2 - 1/2)
        
        # Convert mGal to uGal
        self.g_ugal = self.data['gravity'] * 1e3
        
        # Constants
        self.k = 0.3
        self.h = 0.6
        self.d = 1.53 # delta
    
    def read_file(self) -> pd.DataFrame:
        '''
        Read file containing gravity data
        
        Returns
        -------
        df        : Pandas DataFrame
        '''
        
        column_mapping = {
            'lon': ['lon', 'long', 'longitude', 'x'],
            'lat': ['lat', 'lati', 'latitude', 'y'],
            'h': ['h', 'height', 'z', 'elevation', 'elev'],
            'gravity': ['gravity', 'g', 'acceleration', 'grav']
        }
        file_path = self.path_to_data
        
        if file_path is None:
            raise ValueError('File path not specified')
        
        # Validate file extension (type)
        file_type = file_path.suffix[1:]
        if file_type not in self.VALID_FILE_TYPES:
            raise ValueError(f'Unsupported file format: {file_type}. Supported types: {self.VALID_FILE_TYPES}')
        
        file_reader = {
            'csv' : pd.read_csv,
            'xlsx': pd.read_excel,
            'xls' : pd.read_excel,
            'txt' : lambda filepath: pd.read_csv(filepath, delimiter='\t')
        }
        # Read data
        df = file_reader[file_type](file_path)
        # Rename columns to standardized names
        df = df.rename(columns=lambda col: next((key for key, values in column_mapping.items() if col.lower() in values), col))
        
        return df

    def mean2free(self) -> pd.DataFrame:
        '''
        Convert gravity data in mean tide system to tide-free system
        
        References
        ----------
        1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
                                heights and different permanent tide systems—A case study of New Zealand
                                http://link.springer.com/10.1007/s12518-010-0038-5 (Equations 19-22)
                                
        2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
                                http://www.springerlink.com/index/10.1007/BF02520477 (Page 284; Equations 9-11)
        
        Notes
        -----
        1. We assume gravity units are mGal
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_free = self.g_ugal - self.d * self.g_bracket
        g_free = g_free * 1e-3 # mGal
        
        #  Convert elevation to tide-free system
        h_free = data['h'] + (1 + self.k - self.h) * self.h_bracket
        
        # Update dataframe
        data['g_free'] = g_free
        data['H_free'] = h_free
        
        return data
    
    def free2mean(self) -> pd.DataFrame:
        '''
        Convert gravity data in tide-free system to mean tide system
        
        References
        ----------
        1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
                                heights and different permanent tide systems—A case study of New Zealand
                                http://link.springer.com/10.1007/s12518-010-0038-5 (Equations 19-22)
                                
        2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
                                http://www.springerlink.com/index/10.1007/BF02520477 (Page 284; Equations 9-11)
        
        Notes
        -----
        1. We assume gravity units are mGal
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_mean = self.g_ugal + self.d * self.g_bracket
        g_mean = g_mean * 1e-3 # mGal
        
        #  Convert elevation to tide-free system
        h_mean = data['h'] - (1 + self.k - self.h) * self.h_bracket
        
        # Update dataframe
        data['g_mean'] = g_mean
        data['H_mean'] = h_mean
        
        return data
    
    def zero2free(self) -> pd.DataFrame:
        '''
        Convert gravity data in zero tide to tide-free system
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_free = self.g_ugal - (self.d - 1) * self.g_bracket
        g_free = g_free * 1e-3 # mGal
        
        # Convert height
        h_free = data['h'] + (self.k -  self.h) * self.h_bracket
        
        # Update dataframe
        data['g_free'] = g_free
        data['H_free'] = h_free
        
        return data
    
    def free2zero(self) -> pd.DataFrame:
        '''
        Convert gravity data in tide-free system to zero tide system
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_zero = self.g_ugal + (self.d - 1) * self.g_bracket
        g_zero = g_zero * 1e-3 # mGal
        
        # Convert height
        h_zero = data['h'] - (self.k -  self.h) * self.h_bracket
        
        # Update dataframe
        data['g_zero'] = g_zero
        data['H_zero'] = h_zero
        
        return data
    
    def zero2mean(self) -> pd.DataFrame:
        '''
        Convert gravity data in zero tide to mean tide system
        '''
        data = self.data.copy()
        
        # Convert gravity data to mean tide system
        g_mean = self.g_ugal + self.g_bracket
        g_mean = g_mean * 1e-3 # mGal
        
        # Convert height
        h_mean = data['h'] - self.h_bracket
        
        # Update dataframe
        data['g_mean'] = g_mean
        data['H_mean'] = h_mean
        
        return data
    
    def mean2zero(self) -> pd.DataFrame:
        '''
        Convert gravity data in mean tide to zero tide system
        '''
        data = self.data.copy()
        
        # Convert gravity data to mean tide system
        g_zero = self.g_ugal - self.g_bracket
        g_zero = g_zero * 1e-3 # mGal
        
        # Convert height
        h_zero = data['h'] + self.h_bracket
        
        # Update dataframe
        data['g_zero'] = g_zero
        data['H_zero'] = h_zero
        
        return data
        
    
class GeoidTideSystemConverter:
    '''
    Convert between different permanet tide systems
    
    Parameters
    ----------
    phi               : Geodetic latitude (degrees).
    geoid             : Geoid heights.
    ellipsoid         : reference ellipsoid ('wgs84' or 'grs80').
    
    Parameters
    ----------
    phi               : Geodetic latitude (degrees).
    geoid             : Geoid heights.
    ellipsoid         : reference ellipsoid ('wgs84' or 'grs80').
    '''
    def __init__(self, phi, geoid=None, ellipsoid='wgs84') -> None:
        '''
        Initialize 

        Parameters
        ----------
        phi       : geodetic latitude (degrees)
        geoid     : geoid model (output of ggm.reference_geoid())
        ellipsoid : reference ellipsoid ('wgs84' or 'grs80')
        '''
        self.phi   = phi
        self.geoid = geoid
        self.ellipsoid = constants.wgs84() if 'wgs84' in ellipsoid.lower() else constants.grs80()
        self.semi_major = self.ellipsoid['semi_major']
        self.semi_minor = self.ellipsoid['semi_minor']
        self.varphi = geodetic2geocentric(self.phi, self.semi_major, self.semi_minor)

    def geoid_mean2zero(self) -> np.ndarray:
        '''
        Convert geoid in mean tide system to zero tide system
        
        Returns
        -------
        Nzero     : numpy array of geoid heights in zero tide system (m)
        '''
        if self.geoid is None:
            raise ValueError('Please provide geoid heights.')
        return self.geoid - ( -0.198 * (3/2 * np.sin(np.radians(self.varphi))**2 - 1/2) )
        
    def geoid_mean2free(self) -> np.ndarray:
        '''
        Convert geoid in mean tide system to free tide system
        
        Returns
        -------
        Nfree     : numpy array of geoid heights in free tide system (m)
        '''
        if self.geoid is None:
            raise ValueError('Please provide geoid heights.')
        k = 0.3 # Love number
        return self.geoid - ( (1+k) * (-0.198) * (3/2 * np.sin(np.radians(self.varphi))**2 - 1/2) )
