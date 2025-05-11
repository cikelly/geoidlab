############################################################
# Gravity Reduction CLI interface                          #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import argparse
import sys
import pandas as pd
import numpy as np

from pathlib import Path

from geoidlab.cli.commands.utils.common import directory_setup
from geoidlab.utils.interpolators import Interpolators
from geoidlab import gravity

def anomalies() -> None:
    '''
    Compute gravity anomalies (Free-air and Bouguer)
    '''
    pass

def grid() -> None:
    '''
    '''
    pass

def faye() -> None:
    pass

def main() -> None:
    '''
    Main function for gravity reductions. The supported methods are Free-air and Bouguer reductions,
    with outputs including Free-air and Bouguer anomalies, and Helmert/Faye anomalies
    '''
    
    parser = argparse.ArgumentParser(
        description=(
            'Perform gravity reduction'
        )
    )
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input file with lon, lat, gravity, and height data (required)')
    parser.add_argument('--marine-data', type=str, default=None,
                        help='Input file with lon, lat, gravity, and height data for marine gravity anomalies (optional)')
    parser.add_argument('--do', type=str, default='free-air', choices=['free-air', 'bouguer', 'faye', 'all'],
                        help='Computation steps to perform: [free-air, bouguer, faye, or all (default: free-air)]')
    parser.add_argument('--start', type=str, default=None,
                        help='Start processing from this step: [free-air, bouguer, faye]')
    parser.add_argument('--end', type=str, default=None,
                        help='End processing at this step: [free-air, bouguer, faye]')
    parser.add_argument('--gravity-tide', type=str, default=None, 
                        help='Tide system of the surface gravity data (required for gravity-anomaly): Options: [mean_tide, zero_tide, tide_free]')
    parser.add_argument('--grid', action='store_true', default=False,
                        help='Grid the gravity anomalies over a bounding box')
    parser.add_argument('--grid-size', type=float, default=None, 
                        help='Grid size in minutes (e.g., 5 for a 5-by-5 minute grid). Required if --grid')
    parser.add_argument('--bbox', type=list, default=[None, None, None, None], 
                        help='The bounding box [W,S,E,N] of the study area. Required if --grid')
    parser.add_argument('--bbox-offset', type=float, default=1.0, 
                        help='Offset around the bounding box [W,S,E,N] over which to grid the gravity anomalies. Required if --grid')
    parser.add_argument('--ellipsoid', type=str, default='wgs84', 
                        help='Reference ellipsoid. Supported: [wgs84, grs80]')
    parser.add_argument('--proj-name', type=str, default='GeoidProject', 
                        help='Project directory where downloads and results subdirectories are created')
    parser.add_argument('--converted', action='store_true', default=False,
                        help='Indicate that input data is already in the target tide system')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir) if args.model_dir is not None else Path(args.proj_name) / 'downloads'
    output_dir = Path(args.proj_name) / 'results'
    
    # Set up directory structure
    directory_setup(args.proj_name)


if __name__ == '__main__':
    sys.exit(main())