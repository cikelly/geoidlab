############################################################
# Utilities for converting between tide systems            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import pandas as pd
import numpy as np

class TideSystemConverter:
    '''
    Convert between different permanet tide systems
    
    Parameters
    ----------
    path_to_data      : path to data file
    data              : numpy array or Pandas DataFrame or dict
                        (lon, lat, elevation, gravity)
    
    Methods
    -------
    read_file         : Reads the file containing gravity data and returns 
                        a Pandas DataFrame
    gravity_mean2free : Convert gravity data in mean tide to free tide system
    
    Notes
    -----
    1. Arrange your data in the order: lon, lat, elevation, gravity
    2. Supported file formats: .csv, .txt, .xlsx, .xls
    3. If both path_to_data and data are provided, data will be used
    '''
    def __init__(self, path_to_data=None, data=None):
        self.path_to_data = path_to_data
        # self.data = data if data is not None else self.read_file()
        self.data = data
    
    def read_file(self):
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
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError('Unsupported file format')

        # Rename columns to standardized names
        df = df.rename(columns=lambda col: next((key for key, values in column_mapping.items() if col.lower() in values), col))
        
        return df

    def gravity_mean2free(self):
        '''
        Convert gravity data in mean tide system to free tide system
        
        References
        ----------
        1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
                                heights and different permanent tide systems—A case study of New Zealand
                                http://link.springer.com/10.1007/s12518-010-0038-5
                                
        2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
                                http://www.springerlink.com/index/10.1007/BF02520477
        '''
        # Code for conversion goes here
        if self.data is None and self.path_to_data is None:
            raise ValueError('Provide Data or path to data.')

        if self.data is not None:
            print('Data provided. Converting to free tide system.')
            data = self.data
            
            if isinstance(data, pd.DataFrame):
                # Rename columns to lon, lat, h, and gravity
                data.columns = ['lon', 'lat', 'h', 'gravity']
            elif isinstance(data, np.ndarray):
                # Create a Pandas DataFrame from the numpy array
                data = pd.DataFrame(data, columns=['lon', 'lat', 'h', 'gravity'])
            elif isinstance(data, dict):
                # Create a Pandas DataFrame from the dictionary
                data = pd.DataFrame(data)
                data.columns = ['lon', 'lat', 'h', 'gravity']     
        elif self.path_to_data is not None:
            print('Path to data provided. Reading data from file.')
            # Read data from file
            data = self.read_file()
        
        # Convert gravity data to free tide system
        d = 1.53
        data['gravity_free'] = data['gravity'] * 1000 - d * (-30.4 + 91.2 * np.sin(np.radians(data['lat'])) ** 2) # uGal
        data['gravity_free'] = data['gravity_free'] * 1e-3 # mGal
        # Convert elevation data to free tide system
        k = 0.3
        hc = 0.6
        data['h_free'] = data['h'] + (1 + k - hc) * (-0.198 * (3/2 * np.sin(np.radians(data['lat'])) ** 2 - 1/2)) # m
        
        return data
    
    # TO DO: Add other methods for other tide systems
    
    
    # def geoid_mean2free(self):
    #     '''
    #     Convert free tide to mean tide system for geoid data
        
    #     References
    #     ----------
    #     1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
    #                             heights and different permanent tide systems—A case study of New Zealand
    #                             http://link.springer.com/10.1007/s12518-010-0038-5
                                
    #     2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
    #                             http://www.springerlink.com/index/10.1007/BF02520477
    #     '''
    #     # Code for conversion goes here
    #     pass
    
    # def gravity_free2mean(self):
    #     '''
    #     Convert free tide to mean tide system for gravity data
        
    #     References
    #     ----------
    #     1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
    #                             heights and different permanent tide systems—A case study of New Zealand
    #                             http://link.springer.com/10.1007/s12518-010-0038-5
                                
    #     2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
    #                             http://www.springerlink.com/index/10.1007/BF02520477
    #     '''
    #     # Code for conversion goes here
    #     pass
    
    # def geoid_free2mean(self):
    #     '''
    #     Convert free tide to mean tide system for geoid data
        
    #     References
    #     ----------
    #     1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
    #                             heights and different permanent tide systems—A case study of New Zealand
    #                             http://link.springer.com/10.1007/s12518-010-0038-5
                                
    #     2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
    #                             http://www.springerlink.com/index/10.1007/BF02520477
    #     '''
    #     # Code for conversion goes here
    #     pass
