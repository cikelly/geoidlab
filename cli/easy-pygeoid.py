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

def gravity_field_reduction(bbox, gravity_data, dem=None):
    # Implement gravity field reduction (free-air)
    pass

def residual_gravity_anomalies(bbox, gravity_data, dem=None):
    # Implement residual gravity anomalies computation
    pass

def compute_step(bbox, gravity_data, dem=None):
    # Implement main computation step
    pass

def restore_step(bbox, gravity_data, dem=None):
    # Implement restore step
    pass

def main():
    parser = argparse.ArgumentParser(
        description='Compute gravimetric geoid using provided DEM and gravity data.',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80)
    )
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands for each step of geoid computation')

    # Subparser for compute_geoid
    parser_compute = subparsers.add_parser('compute_geoid', help='Compute gravimetric geoid')
    parser_compute.add_argument('--bbox', '-b', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), required=True, help='Bounding box of area of interest [left (W), bottom (S), right (E), top (N)]')
    parser_compute.add_argument('--gravity', '-g', required=True, help='Path to the gravity data file')
    parser_compute.add_argument('--dem', '-d', required=False, help='Path to the DEM file')
    parser_compute.add_argument('--start-step', '-s', required=False, help='Start step for gravity anomalies')
    parser_compute.add_argument('--end-step', '-e', required=False, help='End step for geoid computation')

    # Subparser for gravity_field_reduction
    parser_reduction = subparsers.add_parser('gravity_field_reduction', help='Perform gravity field reduction (free-air)')
    parser_reduction.add_argument('--bbox', '-b', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), required=True, help='Bounding box of area of interest [left (W), bottom (S), right (E), top (N)]')
    parser_reduction.add_argument('--gravity', '-g', required=True, help='Path to the gravity data file')
    parser_reduction.add_argument('--dem', '-d', required=False, help='Path to the DEM file')

    # Subparser for residual_gravity_anomalies
    parser_residual = subparsers.add_parser('residual_gravity_anomalies', help='Compute residual gravity anomalies')
    parser_residual.add_argument('--bbox', '-b', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), required=True, help='Bounding box of area of interest [left (W), bottom (S), right (E), top (N)]')
    parser_residual.add_argument('--gravity', '-g', required=True, help='Path to the gravity data file')
    parser_residual.add_argument('--dem', '-d', required=False, help='Path to the DEM file')

    # Subparser for compute_step
    parser_compute_step = subparsers.add_parser('compute_step', help='Perform main computation step')
    parser_compute_step.add_argument('--bbox', '-b', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), required=True, help='Bounding box of area of interest [left (W), bottom (S), right (E), top (N)]')
    parser_compute_step.add_argument('--gravity', '-g', required=True, help='Path to the gravity data file')
    parser_compute_step.add_argument('--dem', '-d', required=False, help='Path to the DEM file')

    # Subparser for restore_step
    parser_restore = subparsers.add_parser('restore_step', help='Perform restore step')
    parser_restore.add_argument('--bbox', '-b', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'), required=True, help='Bounding box of area of interest [left (W), bottom (S), right (E), top (N)]')
    parser_restore.add_argument('--gravity', '-g', required=True, help='Path to the gravity data file')
    parser_restore.add_argument('--dem', '-d', required=False, help='Path to the DEM file')

    args = parser.parse_args()

    if args.command == 'compute_geoid':
        compute_geoid(args.bbox, args.gravity, args.dem, args.start_step, args.end_step)
    elif args.command == 'gravity_field_reduction':
        gravity_field_reduction(args.bbox, args.gravity, args.dem)
    elif args.command == 'residual_gravity_anomalies':
        residual_gravity_anomalies(args.bbox, args.gravity, args.dem)
    elif args.command == 'compute_step':
        compute_step(args.bbox, args.gravity, args.dem)
    elif args.command == 'restore_step':
        restore_step(args.bbox, args.gravity, args.dem)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
