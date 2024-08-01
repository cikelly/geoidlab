############################################################
# Utilities for converting between tide systems            #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import pandas as pd

class TideSystemConverter:
    def __init__(self, path_to_data=None, data=None):
        self.path_to_data = path_to_data
        self.data = data if data is not None else self.read_file(path_to_data)
    
    def read_file(file_path):
        '''
        Read file containing gravity data
        
        Parameters
        ----------
        file_path : Path to the file
        
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
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError('Unsupported file format')
    
    def gravity_free2mean(self):
        '''
        Convert free tide to mean tide system for gravity data
        
        References
        ----------
        1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
                                heights and different permanent tide systems—A case study of New Zealand
                                http://link.springer.com/10.1007/s12518-010-0038-5
                                
        2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
                                http://www.springerlink.com/index/10.1007/BF02520477
        '''
        # Code for conversion goes here
        pass
    
    def geoid_free2mean(self):
        '''
        Convert free tide to mean tide system for geoid data
        
        References
        ----------
        1. Tenzer et al. (2010): Assessment of the LVD offsets for the normal-orthometric 
                                heights and different permanent tide systems—A case study of New Zealand
                                http://link.springer.com/10.1007/s12518-010-0038-5
                                
        2. Ekman (1989)        : Impacts of geodynamic phenomena on systems for height and gravity
                                http://www.springerlink.com/index/10.1007/BF02520477
        '''
        # Code for conversion goes here
        pass
    

    

    
    # Update column names to lon, lat, H, and gravity
# def read_file(file_path):
#     # Define possible column names for each required column
#     column_mapping = {
#         'lon': ['lon', 'x'],
#         'lat': ['lat', 'y'],
#         'h': ['h', 'height', 'z'],
#         'gravity': ['gravity', 'g', 'acceleration']
#     }
    
#     # Read the file based on its extension
#     if file_path.endswith('.csv'):
#         df = pd.read_csv(file_path)
#     elif file_path.endswith('.xlsx'):
#         df = pd.read_excel(file_path)
#     elif file_path.endswith('.txt'):
#         df = pd.read_csv(file_path, delimiter='\t')
#     else:
#         raise ValueError('Unsupported file format')
    
#     # Create a new DataFrame with standardized column names
#     standardized_df = pd.DataFrame()
    
#     for standard_col, possible_cols in column_mapping.items():
#         for col in possible_cols:
#             if col in df.columns:
#                 standardized_df[standard_col] = df[col]
#                 break
#         else:
#             raise ValueError(f"Required column '{standard_col}' not found in the file")
    
#     return standardized_df
