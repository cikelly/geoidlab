#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_path))

from easy_pygeoid.terrain import TerrainQuantities
from easy_pygeoid.dem import dem4geoid

def main():
    parser = argparse.ArgumentParser(
        description='Compute terrain quantities (terrain correction and indirect effect)',
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
        '--output-tc',
        default='terrain_correction.nc',
        help='Output file for terrain correction'
    )
    parser.add_argument(
        '--output-ind',
        default='indirect_effect.nc',
        help='Output file for indirect effect'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=110,
        help='Radius for terrain correction in km'
    )
    parser.add_argument(
        '--bbox-off',
        type=float,
        default=1.0,
        help='Buffer around bbox in degrees'
    )
    
    args = parser.parse_args()
    
    # Get or download DEM
    dem = args.dem
    if dem is None:
        print(f'No DEM provided. Downloading SRTM30Plus over {args.bbox}...')
        dem = dem4geoid(
            bbox=args.bbox,
            downloads_dir=args.downloads_dir
        )
    
    # Initialize TerrainQuantities
    print('Initializing terrain computations...')
    tq = TerrainQuantities(
        ori_topo=dem,
        ellipsoid=args.ellipsoid,
        radius=args.radius,
        bbox_off=args.bbox_off
    )
    
    # Compute terrain correction
    print('Computing terrain correction...')
    tc = tq.terrain_correction()
    tc.to_netcdf(args.output_tc)
    print(f'Terrain correction saved to {args.output_tc}')
    
    # Compute indirect effect
    print('Computing indirect effect...')
    ind, _ = tq.indirect_effect()
    ind.to_netcdf(args.output_ind)
    print(f'Indirect effect saved to {args.output_ind}')

if __name__ == '__main__':
    main()