#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import xarray as xr
import numpy as np
# Get the absolute path to the src directory
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_path))

from geoidlab.dem import dem4geoid
from geoidlab.geoid import ResidualGeoid
from geoidlab.ggm_tools import GlobalGeopotentialModel
from geoidlab.terrain import TerrainQuantities
from geoidlab.tide import GravityTideSystemConverter
from geoidlab.utils.interpolators import Interpolators

from typing import Tuple

def run_remove_step(args) -> Tuple[np.ndarray, GlobalGeopotentialModel]:
    '''Handle the remove step - compute residual gravity anomalies'''
    # Download DEM if needed
    dem = args.dem
    if dem is None:
        print(f'No DEM provided. Downloading SRTM30Plus over {args.bbox}...')
        try:
            dem = dem4geoid(bbox=args.bbox, downloads_dir='downloads')
        except Exception as e:
            print(f'Download failed. Error: {e}.')
            return None
            
    # Convert gravity data to tide-free system if needed
    grav_conv = GravityTideSystemConverter(path_to_data=args.gravity)
    grav_data = grav_conv.gravity_mean2free()
    
    # Calculate terrain correction
    tq = TerrainQuantities(
        ori_topo=dem, 
        ellipsoid=args.ellipsoid,
        radius=110,
        bbox_off=1
    )
    tc = tq.terrain_correction()
    
    # Calculate residual anomalies 
    ggm = GlobalGeopotentialModel(
        model_name=args.ggm,
        grav_data=grav_data,
        ellipsoid=args.ellipsoid,
        nmax=args.nmax,
        zonal_harmonics=True,
        model_dir='downloads'
    )
    
    return tc, ggm

def run_compute_step(args, tc=None, ggm=None) -> Tuple[np.ndarray, np.ndarray]:
    '''Handle the compute step - calculate residual and reference geoids'''
    if tc is None or ggm is None:
        print("Remove step must be run first")
        return None
        
    # Calculate residual geoid
    res_geoid = ResidualGeoid(
        res_anomaly=tc,
        method=args.stokes_method,
        ellipsoid=args.ellipsoid,
        sph_cap=180/args.nmax,
        sub_grid=args.bbox,
        nmax=args.nmax
    )
    N_res = res_geoid.compute_geoid()
    
    # Calculate reference geoid
    N_ref = ggm.geoid(icgem=True)
    
    return N_res, N_ref

def run_restore_step(args, N_res=None, N_ref=None, tc=None) -> xr.Dataset:
    """Handle the restore step - combine geoid components"""
    if N_res is None or N_ref is None or tc is None:
        print("Compute step must be run first")
        return None
    
    # Calculate indirect effect
    tq = TerrainQuantities(
        ori_topo=tc, 
        ellipsoid=args.ellipsoid,
        radius=110,
        bbox_off=1
    )
    N_ind, _ = tq.indirect_effect()
    
    # Combine components
    N = N_res + N_ref + N_ind
    return N
    
def main() -> None:
    '''Main function for command line interface'''
    parser = argparse.ArgumentParser(
        description='Compute gravimetric geoid using Remove-Compute-Restore method',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80)
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Steps in RCR process')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--bbox', '-b', 
        type=float, nargs=4, 
        metavar=('W', 'S', 'E', 'N'),
        required=True,
        help='Bounding box [West, South, East, North]'
    )
    common_parser.add_argument(
        '--gravity', '-g',
        required=True,
        help='Path to gravity data file'
    )
    common_parser.add_argument(
        '--dem', '-d',
        help='Optional path to DEM file'
    )
    common_parser.add_argument(
        '--ggm',
        default='GO_CONS_GCF_2_DIR_R6',
        help='Global Geopotential Model to use'
    )
    common_parser.add_argument(
        '--ellipsoid',
        default='grs80',
        choices=['grs80', 'wgs84'],
        help='Reference ellipsoid'
    )
    common_parser.add_argument(
        '--nmax',
        type=int,
        default=222,
        help='Maximum spherical harmonic degree'  
    )

    # Remove step
    remove_parser = subparsers.add_parser(
        'remove',
        parents=[common_parser],
        help='Remove step - compute residual gravity anomalies'
    )

    # Compute step  
    compute_parser = subparsers.add_parser(
        'compute',
        parents=[common_parser], 
        help='Compute step - calculate residual and reference geoids'
    )
    compute_parser.add_argument(
        '--stokes-method',
        choices=['hg', 'wg', 'og', 'ml'],
        default='wg',
        help='Stokes kernel modification method'
    )

    # Restore step
    restore_parser = subparsers.add_parser(
        'restore',
        parents=[common_parser],
        help='Restore step - combine geoid components'
    )

    # Full pipeline
    pipeline_parser = subparsers.add_parser(
        'compute-geoid',
        parents=[common_parser],
        help='Run full RCR pipeline'
    )
    pipeline_parser.add_argument(
        '--stokes-method',
        choices=['hg', 'wg', 'og', 'ml'],
        default='wg', 
        help='Stokes kernel modification method'
    )

    args = parser.parse_args()

    if args.command == 'remove':
        run_remove_step(args)
    elif args.command == 'compute':
        tc, ggm = run_remove_step(args)
        run_compute_step(args, tc, ggm)
    elif args.command == 'restore':
        tc, ggm = run_remove_step(args)
        N_res, N_ref = run_compute_step(args, tc, ggm)
        run_restore_step(args, N_res, N_ref, tc)
    elif args.command == 'compute-geoid':
        tc, ggm = run_remove_step(args)
        N_res, N_ref = run_compute_step(args, tc, ggm)
        N = run_restore_step(args, N_res, N_ref, tc)
        return N
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
