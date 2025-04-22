#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_path))

from easy_pygeoid.dem import dem4geoid
from easy_pygeoid.ggm_tools import GlobalGeopotentialModel

def main():
    parser = argparse.ArgumentParser(
        description='Download GGM and DEM data required for geoid computation',
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
        '--ggm',
        default='GO_CONS_GCF_2_DIR_R6',
        help='Global Geopotential Model to download'
    )
    parser.add_argument(
        '--downloads-dir',
        default='downloads',
        help='Directory to save downloaded files'
    )
    parser.add_argument(
        '--dem-resolution',
        type=int,
        default=30,
        help='DEM resolution in arc-seconds'
    )
    
    args = parser.parse_args()
    
    # Create downloads directory
    downloads_dir = Path(args.downloads_dir)
    downloads_dir.mkdir(exist_ok=True)
    
    print(f'Downloading DEM for region {args.bbox}...')
    dem = dem4geoid(
        bbox=args.bbox,
        downloads_dir=args.downloads_dir,
        resolution=args.dem_resolution
    )
    
    print(f'Downloading {args.ggm}...')
    GlobalGeopotentialModel(
        model_name=args.ggm,
        model_dir=args.downloads_dir
    )
    
    print('Downloads complete!')

if __name__ == '__main__':
    main()