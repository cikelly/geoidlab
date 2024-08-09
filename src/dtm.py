############################################################
# Utilities for calculating reference geoid                #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import lzma
import os

import numpy as np

# from tqdm import tqdm

class DigitalTerrainModel:
    def __init__(self, model_name=None, nmax=2190):
        self.name = model_name
        self.nmax = nmax
        
        if self.name is None:
            script_dir = os.path.dirname(__file__)
            self.name = os.path.join(script_dir, 'data', 'DTM2006.xz')
            print(f'Using compressed file in src/data directory ...')
            with lzma.open(self.name, 'rt') as f:
                self.dtm = f.readlines()
        else:
            with open(self.name, 'r') as f:
                self.dtm = f.readlines()
            
    def read_dtm2006(self):
        '''
        Read DTM data stored as compressed LZMA file or the original DTM2006 file
        '''

        HCnm = np.zeros((self.nmax+1, self.nmax+1))
        HSnm = np.zeros((self.nmax+1, self.nmax+1))
            
            
        for line in self.dtm:
            line = line.split()
            
            n = int(line[0])
            m = int(line[1])

            if n > self.nmax:
                break
            
            if n <= self.nmax+1 and m <= self.nmax+1:
                HCnm[n,m] = float(line[2].replace('D', 'E'))
                HSnm[n,m] = float(line[3].replace('D', 'E'))
            
        return self.HCnm, self.HSnm

    def calculate_height(self):
        pass