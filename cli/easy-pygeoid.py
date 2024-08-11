#!/usr/bin/env python3
import argparse
from src.dem import dem4geoid
# import matplotlib.pyplot as plt

def compute_geoid(bbox, gravity_data, dem=None, start=None, end=None):
    # Download DEM if not provided
    if dem is None:
        print(f'No DEM provided. Downloading SRTM30Plus over {bbox}...')
        try:
            dem = dem4geoid(bbox=bbox, downloads_dir='downloads')
            # dem['z'].plot(cmap='terrain')
            # plt.show()
        except Exception as e:
            print(f'Download failed. Error: {e}.')
            return  
            
    # Compute geoid


    
def main():
    parser = argparse.ArgumentParser(
        description='Compute gravimetric geoid using provided DEM and gravity data.',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80)
    )
    parser.add_argument(
        '--bbox', '-b', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), 
        required=True, help='Bounding box of area of interest [left (W), bottom (S), right (E), top (N)]'
    )
    parser.add_argument('--gravity', '-g', required=True, help='Path to the gravity data file')
    parser.add_argument('--dem', '-d', required=False, help='Path to the DEM file')
    parser.add_argument('--start-step', '-s', required=False, help='Start step for gravity anomalies')
    parser.add_argument('--end-step', '-e', required=False, help='End step for geoid computation')
    

    args = parser.parse_args()
    compute_geoid(args.bbox, args.gravity, args.dem, args.start_step, args.end_step)

if __name__ == "__main__":
    main()
