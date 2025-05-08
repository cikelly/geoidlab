############################################################
# Utilities for converting between tide systems            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path

# from geoidlab import constants
# from geoidlab.coordinates import geodetic2geocentric

@staticmethod
def m_s2_to_mgal(gravity_m_s2: float) -> float:
    return gravity_m_s2 * 100_000  # 1 m/s² = 100,000 mGal

@staticmethod
def gal_to_mgal(gravity_gal: float) -> float:
    return gravity_gal * 1_000  # 1 Gal = 1,000 mGal

class GravityTideSystemConverter:
    '''
    Convert between different permanet tide systems for gravity and physical height
    
    Parameters
    ----------
    path_to_data: path to data file
    data        : numpy array or Pandas DataFrame or dict
                  (lon, lat, elevation, gravity)
    k           : Love number (default: 0.3)
    h           : Love number (default: 0.6)
    d           : Delta factor (default: 1.53)
    
    Notes
    -----
    - Arrange your data in the order: lon, lat, elevation (m), gravity (mGal)
    - If both path_to_data and data are provided, data will be used
    
    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'lon': [0, 1], 'lat': [45, 46], 'H': [100, 200], 'g': [980_000, 980_001]})
    >>> converter = GravityTideSystemConverter(data=data)
    >>> result = converter.mean2free()
    >>> print(result)
    '''
    VALID_FILE_TYPES = ['csv', 'txt', 'xlsx', 'xls']
    
    def __init__(
        self, 
        path_to_data: str = None, 
        data: np.ndarray | pd.DataFrame | dict = None,
        k: float = 0.3,
        h: float = 0.6,
        d: float = 1.53
    ) -> None:
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
        self.k = k
        self.h = h
        self.d = d
        
        # Read or process data
        if self.path_to_data is not None:
            self.data = self.read_file()
        else:
            if isinstance(data, pd.DataFrame):
                self.data = data.copy()
                self.data.columns = ['lon', 'lat', 'H', 'gravity']
            elif isinstance(data, np.ndarray):
                self.data = pd.DataFrame(self.data, columns=['lon', 'lat', 'H', 'gravity'])
            elif isinstance(data, dict):
                self.data = pd.DataFrame(self.data)
                self.data.columns = ['lon', 'lat', 'H', 'gravity']
            else:
                raise ValueError('Data must be a Pandas DataFrame, numpy array, or dictionary.')
        
        # Validate data contents
        required_columns = ['lon', 'lat', 'H', 'gravity']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f'Data must contain columns: {required_columns}.')
        
        # Validate ranges
        if not (self.data['lat'].between(-90, 90).all()):
            raise ValueError('Latitude must be between -90 and 90 degrees.')
        if not (self.data['lon'].between(-180, 180).all()):
            raise ValueError('Longitude must be between -180 and 180 degrees.')
        if not (self.data['gravity'].between(974_000, 984_000)).all():
            raise ValueError('Gravity values must be between 974,000 and 984,000 mGal. Ensure values are in mGal and physically plausible.')
        
        
        # Precompute terms in bracket
        SIN2PHI = np.sin(np.radians(self.data['lat'])) ** 2
        self.g_bracket = -30.4 + 91.2 * SIN2PHI # uGal
        self.H_bracket = -0.198 * (3/2 * SIN2PHI - 1/2) # m
        
        # Convert mGal to uGal
        self.g_ugal = self.data['gravity'] * 1e3
    
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
            'H': ['H', 'height', 'z', 'elevation', 'elev'],
            'gravity': ['gravity', 'g', 'acceleration', 'grav']
        }
        
        if self.file_path is None:
            raise ValueError('File path not specified')
        
        # Validate file extension (type)
        file_type = self.file_path.suffix[1:].lower()
        if file_type not in self.VALID_FILE_TYPES:
            raise ValueError(f'Unsupported file format: {file_type}. Supported types: {self.VALID_FILE_TYPES}')
        
        file_reader = {
            'csv' : pd.read_csv,
            'xlsx': pd.read_excel,
            'xls' : pd.read_excel,
            'txt' : lambda filepath: pd.read_csv(filepath, delimiter='\t')
        }
        # Read data
        df = file_reader[file_type](self.file_path)
        # Rename columns to standardized names
        df = df.rename(columns=lambda col: next((key for key, values in column_mapping.items() if col.lower() in values), col))
        
        # required_columns = ['lon', 'lat', 'H', 'gravity']
        if not all(col in df.columns for col in column_mapping.keys()):
            raise ValueError(f'File must contain columns for: {column_mapping.keys()}. Found: {list(df.columns)}')
        
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
        2. g_free = g_mean - d * (-30.4 + 91.2 * sin²(lat)) [µGal]
        3. H_free = H_mean + (1 + k - h) * (-0.198 * (3/2 * sin²(lat) - 1/2)) [m]

        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_free = (self.g_ugal - self.d * self.g_bracket) * 1e-3 # mGal
        
        #  Convert elevation to tide-free system
        H_free = data['H'] + (1 + self.k - self.h) * self.H_bracket
        
        # Update dataframe
        data['g_free'] = g_free
        data['H_free'] = H_free
        
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
        2. g_mean = g_free + d * (-30.4 + 91.2 * sin²(lat)) [µGal]
        3. H_mean = H_free - (1 + k - h) * (-0.198 * (3/2 * sin²(lat) - 1/2)) [m]
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_mean = (self.g_ugal + self.d * self.g_bracket) * 1e-3
        
        #  Convert elevation to tide-free system
        H_mean = data['H'] - (1 + self.k - self.h) * self.H_bracket
        
        # Update dataframe
        data['g_mean'] = g_mean
        data['H_mean'] = H_mean
        
        return data
    
    def zero2free(self) -> pd.DataFrame:
        '''
        Convert gravity data in zero tide to tide-free system
        
        Notes
        -----
        1. g_free = g_zero - (d - 1) * (-30.4 + 91.2 * sin²(lat)) [µGal]
        2. H_free = H_zero + (k - h) * (-0.198 * (3/2 * sin²(lat) - 1/2)) [m]
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_free = (self.g_ugal - (self.d - 1) * self.g_bracket) * 1e-3 # mGal
                
        # Convert height
        H_free = data['H'] + (self.k -  self.h) * self.H_bracket
        
        # Update dataframe
        data['g_free'] = g_free
        data['H_free'] = H_free
        
        return data
    
    def free2zero(self) -> pd.DataFrame:
        '''
        Convert gravity data in tide-free system to zero tide system
        
        Notes
        -----
        1. g_zero = g_free + (d - 1) * (-30.4 + 91.2 * sin²(lat)) [µGal]
        2. H_zero = H_free - (k - h) * (-0.198 * (3/2 * sin²(lat) - 1/2)) [m]
        '''
        data = self.data.copy()
        
        # Convert gravity data to tide-free system
        g_zero = (self.g_ugal + (self.d - 1) * self.g_bracket) * 1e-3 # mGal
        
        # Convert height
        H_zero = data['H'] - (self.k -  self.h) * self.H_bracket
        
        # Update dataframe
        data['g_zero'] = g_zero
        data['H_zero'] = H_zero
        
        return data
    
    def zero2mean(self) -> pd.DataFrame:
        '''
        Convert gravity data in zero tide to mean tide system
        
        Notes
        -----
        1. g_mean = g_zero + (-30.4 + 91.2 * sin²(lat)) [µGal]
        2. H_mean = H_zero - (-0.198 * (3/2 * sin²(lat) - 1/2)) [m]
        '''
        data = self.data.copy()
        
        # Convert gravity data to mean tide system
        g_mean = (self.g_ugal + self.g_bracket) * 1e-3 # mGal
        
        # Convert height
        H_mean = data['H'] - self.H_bracket
        
        # Update dataframe
        data['g_mean'] = g_mean
        data['H_mean'] = H_mean
        
        return data
    
    def mean2zero(self) -> pd.DataFrame:
        '''
        Convert gravity data in mean tide to zero tide system
        
        Notes
        -----
        1. g_zero = g_mean - (-30.4 + 91.2 * sin²(lat)) [µGal]
        2. H_zero = H_mean + (-0.198 * (3/2 * sin²(lat) - 1/2)) [m]
        '''
        data = self.data.copy()
        
        # Convert gravity data to mean tide system
        g_zero = (self.g_ugal - self.g_bracket) * 1e-3 # mGal
        
        # Convert height
        H_zero = data['H'] + self.H_bracket
        
        # Update dataframe
        data['g_zero'] = g_zero
        data['H_zero'] = H_zero
        
        return data
        
    
class GeoidTideSystemConverter:
    '''
    Convert between different permanet tide systems

    References
    ----------
    Rapp (1989): The treatment of permanent tidal effects in the analysis of satellite altimeter data for sea surface topography
                 https://link.springer.com/article/10.1007/BF03655376
                 
    Examples
    --------
    >>> import numpy as np
    >>> phi = np.array([45, 46])
    >>> geoid = np.array([30.0, 30.1])
    >>> converter = GeoidTideSystemConverter(phi, geoid)
    >>> N_zero = converter.mean2zero()
    >>> print(N_zero)
    '''
    
    def __init__(
        self, 
        phi: np.ndarray = None, 
        geoid: np.ndarray | xr.Dataset = None,
        path_to_data: str | Path = None,
        varname: str = 'N',
        k: float = 0.3
    ) -> None:
        '''
        Initialize GeoidTideSystemConverter.

        Parameters
        ----------
        phi          : geodetic latitude in degrees (default: None)
        geoid        : geoid heights
        path_to_data : path to NetCDF file of geoid data
        varname      : variable name for geoid heights in NetCDF file or xarray Dataset
        k            : Tidal Love number (default: 0.3)
        
        Notes
        -----
        - Either `geoid` or `path_to_data` must be provided.
        - If `geoid` is a Numpy array, `phi` must be provided with matching shape
        - NetCDF files and Datasets must contain 'lat', 'lon', and `varname` variables
        '''
        self.k = k
        self.path_to_data = Path(path_to_data) if path_to_data is not None else None
        
        if path_to_data is not None:
            try:
                ds = xr.open_dataset(path_to_data)
            except Exception as e:
                raise ValueError(f'Failed to read NetCDF file {path_to_data}: {str(e)}')
            if varname not in ds or 'lon' not in ds or 'lat' not in ds:
                raise ValueError(f'NetCDF file must contain variables: lon, lat, and {varname}.')
            _, self.phi = np.meshgrid(ds['lon'].values, ds['lat'].values)
            self.geoid = ds[varname].values
        elif geoid is None:
            raise ValueError('Please provide geoid heights or path to NetCDF file.')
        elif isinstance(geoid, xr.Dataset):
            if varname not in geoid or 'lon' not in geoid or 'lat' not in geoid:
                raise ValueError(f'xarray Dataset must contain variables: lon, lat, and {varname}.')
            _, self.phi = np.meshgrid(geoid['lon'].values, geoid['lat'].values)
            self.geoid = geoid[varname].values
        elif isinstance(geoid, np.ndarray):
            if phi is None:
                raise ValueError('Provide the latitude grid corresponding to the geoid heights.')
            self.phi   = phi
            self.geoid = geoid
        else:
            raise ValueError('geoid must be a Numpy array or xarray Dataset.')
        
        if not np.all((self.phi >= -90) & (self.phi <= 90)):
            raise ValueError('Latitude must be between -90 and 90 degrees.')
        if self.geoid.shape != self.phi.shape:
            raise ValueError(f'geoid shape {self.geoid.shape} must match phi shape {self.phi.shape}.')
        
        # Compute x and y terms
        self.x = -0.198 # [m]
        self.y = 3/2 * np.sin(np.radians(self.phi))**2 - 1/2


    def mean2zero(self) -> np.ndarray:
        '''
        Convert geoid in mean tide system to zero tide system
        
        Returns
        -------
        N_zero     : numpy array of geoid heights in zero tide system (m)
        
        Notes
        -----
        1. N_zero = N_mean - x * y, where x = -0.198, y = 3/2 * sin²(phi) - 1/2 [m]
        '''
        
        return self.geoid - self.x * self.y
    
    def zero2mean(self) -> np.ndarray:
        '''
        Convert geoid in zero tide system to mean tide system
        
        Returns
        -------
        N_mean     : numpy array of geoid heights in mean tide system (m)
        
        Notes
        -----
        1. N_mean = N_zero + x * y, where x = -0.198, y = 3/2 * sin²(phi) - 1/2 [m]
        '''
        
        return self.geoid + self.x * self.y
        
    def mean2free(self) -> np.ndarray:
        '''
        Convert geoid in mean tide system to tide-free system
        
        Returns
        -------
        N_free     : numpy array of geoid heights in free tide system (m)
        
        Notes
        -----
        1. N_free = N_mean - (1 + k) * x * y, where x = -0.198, y = 3/2 * sin²(phi) - 1/2 [m]
        '''
        
        return self.geoid - ( (1 + self.k) * self.x * self.y )
    
    def free2mean(self) -> np.ndarray:
        '''
        Convert geoid in tide-free system to mean tide system
        
        Returns
        -------
        N_mean     : numpy array of geoid heights in mean tide system (m)
        
        Notes
        -----
        1. N_mean = N_free + (1 + k) * x * y, where x = -0.198, y = 3/2 * sin²(phi) - 1/2 [m]
        '''
        
        return self.geoid + ( (1 + self.k) * self.x * self.y )
    
    def zero2free(self) -> np.ndarray:
        '''
        Convert geoid in zero tide system to tide-free system
        
        Returns
        -------
        N_free     : numpy array of geoid heights in free tide system (m)
        
        Notes
        -----
        1. N_free = N_zero - k * x * y, where x = -0.198, y = 3/2 * sin²(phi) - 1/2 [m]
        '''
        
        return self.geoid - ( self.k * self.x * self.y )
    
    def free2zero(self) -> np.ndarray:
        '''
        Convert geoid in tide-free system to zero tide system
        
        Returns
        -------
        N_zero     : numpy array of geoid heights in zero tide system (m)
        
        Notes
        -----
        1. N_zero = N_free + k * x * y, where x = -0.198, y = 3/2 * sin²(phi) - 1/2 [m]
        '''
        
        return self.geoid + ( self.k * self.x * self.y )
