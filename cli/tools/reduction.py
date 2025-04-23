#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
import pandas as pd

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_path))

from easy_pygeoid import gravity
from easy_pygeoid.terrain import TerrainQuantities
from easy_pygeoid.dem import dem4geoid
from easy_pygeoid.tide import GravityTideSystemConverter

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute gravity reductions and terrain corrections',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80)
    )
    
    parser.add_argument(
        '--gravity', '-g',
        required=True,
        help='Path to gravity data file'
    )
    parser.add_argument(
        '--bbox', '-b', 
        type=float, nargs=4, 
        metavar=('W', 'S', 'E', 'N'),
        required=True,
        help='Bounding box [West, South, East, North]'
    )
    parser.add_argument(
        '--dem', '-d',
        help='Optional path to DEM file'
    )
    parser.add_argument(
        '--ellipsoid',
        default='grs80',
        choices=['grs80', 'wgs84'],
        help='Reference ellipsoid'
    )
    parser.add_argument(
        '--downloads-dir',
        default='downloads',
        help='Directory for downloaded files'
    )
    parser.add_argument(
        '--output', '-o',
        default='gravity_reductions.csv',
        help='Output file for reduced gravity'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=110,
        help='Radius for terrain correction in km'
    )
    parser.add_argument(
        '--tide-system',
        choices=['mean_tide', 'tide_free', 'zero_tide'],
        default='tide_free',
        help='Tide system for gravity values'
    )
    
    args = parser.parse_args()
    
    # Load and convert tide system if needed
    print(f'Converting gravity from mean tide to {args.tide_system}...')
    grav_conv = GravityTideSystemConverter(path_to_data=args.gravity)
    if args.tide_system == 'tide_free':
        df = grav_conv.gravity_mean2free()
    elif args.tide_system == 'zero_tide':
        df = grav_conv.gravity_mean2zero()
    else:
        df = pd.read_csv(args.gravity)
    
    # Calculate free-air anomalies
    print('Computing free-air anomalies...')
    Dg_FA, _ = gravity.gravity_anomalies(
        lat=df['lat'],
        gravity=df['gravity'],
        elevation=df['height'],
        ellipsoid=args.ellipsoid
    )
    
    # Get or download DEM
    dem = args.dem
    if dem is None:
        print(f'No DEM provided. Downloading SRTM30Plus over {args.bbox}...')
        dem = dem4geoid(
            bbox=args.bbox,
            downloads_dir=args.downloads_dir
        )
    
    # Calculate terrain correction
    print('Computing terrain correction...')
    tq = TerrainQuantities(
        ori_topo=dem,
        ellipsoid=args.ellipsoid,
        radius=args.radius,
        bbox_off=1
    )
    tc = tq.terrain_correction()
    
    # Interpolate terrain correction to gravity points
    from scipy.interpolate import RegularGridInterpolator
    lons = tq.LonP[0, :]
    lats = tq.LatP[:, 0]
    interpolator = RegularGridInterpolator(
        (lats, lons),
        tc['tc'].values
    )
    points = df[['lat', 'lon']].values
    Dg_tc = interpolator(points)
    
    # Save results
    output = pd.DataFrame({
        'lon': df['lon'],
        'lat': df['lat'],
        'height': df['height'],
        'gravity': df['gravity'],
        'free_air': Dg_FA,
        'terrain_correction': Dg_tc,
        'helmert': Dg_FA + Dg_tc
    })
    output.to_csv(args.output, index=False)
    print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()