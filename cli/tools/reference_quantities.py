#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_path))

from easy_pygeoid.ggm_tools import GlobalGeopotentialModel

def main():
    parser = argparse.ArgumentParser(
        description='Compute reference quantities (gravity anomalies, geoid) from GGM',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80)
    )
    
    parser.add_argument(
        '--bbox', '-b', 
        type=float, nargs=4, 
        metavar=('W', 'S', 'E', 'N'),
        required=True,
        help='Bounding box [West, South, East, North]'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=5/60,  # 5 arc-minutes
        help='Grid resolution in degrees'
    )
    parser.add_argument(
        '--ggm',
        default='GO_CONS_GCF_2_DIR_R6',
        help='Global Geopotential Model to use'
    )
    parser.add_argument(
        '--ellipsoid',
        default='grs80',
        choices=['grs80', 'wgs84'],
        help='Reference ellipsoid'
    )
    parser.add_argument(
        '--nmax',
        type=int,
        default=222,
        help='Maximum degree of spherical harmonic expansion'
    )
    parser.add_argument(
        '--downloads-dir',
        default='downloads',
        help='Directory containing GGM file'
    )
    parser.add_argument(
        '--output-anomaly',
        default='reference_anomaly.csv',
        help='Output file for reference anomalies'
    )
    parser.add_argument(
        '--output-geoid',
        default='reference_geoid.csv',
        help='Output file for reference geoid heights'
    )
    parser.add_argument(
        '--icgem',
        action='store_true',
        help='Use ICGEM geoid computation'
    )
    
    args = parser.parse_args()
    
    # Create evaluation grid
    minx, maxx = args.bbox[0], args.bbox[2]
    miny, maxy = args.bbox[1], args.bbox[3]
    
    num_x_points = int((maxx - minx) / args.resolution) + 1
    num_y_points = int((maxy - miny) / args.resolution) + 1
    
    lon_grid = np.linspace(minx, maxx, num_x_points)
    lat_grid = np.linspace(miny, maxy, num_y_points)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
    # Prepare data for GGM
    grav_data = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        'height': np.zeros_like(lon_grid.flatten()),
        'gravity': np.zeros_like(lon_grid.flatten())
    })
    
    print(f'Computing reference quantities using {args.ggm} to degree {args.nmax}...')
    ggm = GlobalGeopotentialModel(
        model_name=args.ggm,
        grav_data=grav_data,
        ellipsoid=args.ellipsoid,
        nmax=args.nmax,
        zonal_harmonics=True,
        model_dir=args.downloads_dir
    )
    
    # Compute reference anomalies
    print('Computing reference gravity anomalies...')
    Dg_ref = ggm.gravity_anomaly()
    Dg_ref = Dg_ref.reshape(lon_grid.shape)
    
    # Compute reference geoid
    print('Computing reference geoid...')
    N_ref = ggm.geoid(icgem=args.icgem)
    N_ref = N_ref.reshape(lon_grid.shape)
    
    # Save results
    print('Saving results...')
    results_anomaly = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        'anomaly': Dg_ref.flatten()
    })
    results_anomaly.to_csv(args.output_anomaly, index=False)
    
    results_geoid = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        'geoid_height': N_ref.flatten()
    })
    results_geoid.to_csv(args.output_geoid, index=False)
    
    print(f'Anomalies saved to {args.output_anomaly}')
    print(f'Geoid heights saved to {args.output_geoid}')

if __name__ == '__main__':
    main()