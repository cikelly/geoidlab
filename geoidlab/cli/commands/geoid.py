############################################################
# Geoid  CLI interface                                     #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import argparse
import sys

import xarray as xr

from pathlib import Path

from geoidlab.cli.commands.reference import add_reference_arguments, GGMSynthesis
from geoidlab.cli.commands.faye import add_faye_arguments, GravityReduction
from geoidlab.cli.commands.topo import add_topo_arguments, TopographicQuantities
from geoidlab.cli.commands.utils.common import directory_setup, to_seconds
from geoidlab.geoid import ResidualGeoid
from geoidlab.icgem import get_ggm_tide_system
from geoidlab.utils.io import save_to_netcdf

METHODS_DICT = {
    'hg': "Heck & Gruninger's modification",
    'wg': "Wong & Gore's modification",
    'og': "Original Stokes'",
    'ml': "Meissl's modification",
}
def copy_arguments(src_parser, dst_parser, exclude=None) -> None:
    '''
    Copy arguments from one src_parser to dst_parser, skipping duplicates based on dest
    
    Parameters
    ----------
    src_parser  : Source ArgumentParser with arguments to copy
    dest_parser : Destination ArgumentParser to receive arguments
    exclude     : List of dest names to exclude (optional)
    '''    
    exclude = exclude or []
    existing_dests = {action.dest for action in dst_parser._actions}
    for action in src_parser._actions:
        if action.dest not in existing_dests and action.dest not in exclude:
            kwargs = {
                'dest': action.dest,
                'help': action.help,
                'default': action.default,
            }
            args = action.option_strings if action.option_strings else [action.dest]

            # Handle different action types
            if isinstance(action, argparse._StoreAction):
                kwargs['type'] = action.type
                kwargs['choices'] = action.choices
                kwargs['nargs'] = action.nargs
            elif isinstance(action, argparse._StoreTrueAction):
                kwargs['action'] = 'store_true'
            elif isinstance(action, argparse._StoreFalseAction):
                kwargs['action'] = 'store_false'
            elif isinstance(action, argparse._StoreConstAction):
                kwargs['action'] = 'store_const'
                kwargs['const'] = action.const
            else:
                raise ValueError(f"Unsupported action type for '{action.dest}': {type(action)}")

            dst_parser.add_argument(*args, **kwargs)
            existing_dests.add(action.dest)

def add_geoid_arguments(parser) -> None:
    # parser = argparse.ArgumentParser(description='Geoid computation CLI')
    
    tem_parser_ref = argparse.ArgumentParser(add_help=False)
    add_reference_arguments(tem_parser_ref)
    
    tem_parser_topo = argparse.ArgumentParser(add_help=False)
    add_topo_arguments(tem_parser_topo)
    
    tem_parser_faye = argparse.ArgumentParser(add_help=False)
    add_faye_arguments(tem_parser_faye)
    
    # Arguments to exclude
    exclude_list = ['grid', 'do', 'start', 'end']
    
    # Merge arguments into the main parser
    copy_arguments(tem_parser_ref, parser, exclude=exclude_list)
    copy_arguments(tem_parser_topo, parser, exclude=exclude_list)
    copy_arguments(tem_parser_faye, parser, exclude=exclude_list)
    
    parser.add_argument('--sph-cap', type=float, default=1.0,
                        help='Spherical cap for integration in degrees (default: 1.0)')
    parser.add_argument('--method', type=str, default='hg', choices=['hg', 'wg', 'ml', 'og'],
                        help='Geoid computation method (default: hg). Options: Heck & Gruninger (hg), Wong & Gore (wg), Meissl (ml), original (og)')
    parser.add_argument('--ind-grid-size', type=float, default=30,
                        help='Grid resolution for computing indirect effect. Keep this in seconds. Default: 30 seconds')
    parser.add_argument('--target-tide-system', type=str, default='tide_free', choices=['mean_tide', 'tide_free', 'zero_tide'],
                        help='The tide system that the final geoid should be in. Default: tide_free')
    
def main(args=None) -> None:
    '''
    Main function to handle command line arguments and execute geoid computation.
    '''
    
    if args is None:
        parser = argparse.ArgumentParser(
            description=(
                'Complete workflow for geoid computation using the remove-compute-restore (RCR) method.'
                'Options for solving Stokes\'s integral include Heck & Gruninger (hg), Wong & Gore (wg), Meissl (ml), and original (og).'
            )
        )
        add_geoid_arguments(parser)
        args = parser.parse_args()
    
    # Add geoid-specific arguments
    
    # parser.add_argument('--window-mode', type=str, default='cap', choices=['fixed', 'cap'],
    #                     help='Window mode for geoid computation (default: cap)')
    
    # Set up directories
    directory_setup(args.proj_name)
    model_dir = Path(args.proj_name) / 'downloads'
    output_dir = Path(args.proj_name) / 'results'; 
    
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate and define computation grid
    if not (args.bbox and args.grid_size and args.grid_unit):
        raise ValueError('bbox, grid_size, and grid_unit must be provided for geoid computation')
    bbox = args.bbox
    grid_size = args.grid_size
    grid_unit = args.grid_unit
    
    model_path = (Path(model_dir) / Path(args.model)).with_suffix('.gfc')
    
    ggm_tide = get_ggm_tide_system(icgem_file=model_path, model_dir=model_dir)
    
    # Step 1: Calculate Helmert/Faye anomalies (Dg)
    gravity_reduction = GravityReduction(
        input_file=args.input_file,
        model=args.model,
        model_dir=model_dir,
        marine_data=args.marine_data,
        gravity_tide=args.gravity_tide,
        ellipsoid=args.ellipsoid,
        converted=args.converted,
        grid=True,  # Ensure gridded output
        grid_size=grid_size,
        grid_unit=grid_unit,
        grid_method=args.grid_method,
        bbox=bbox,
        bbox_offset=args.bbox_offset,
        proj_name=args.proj_name,
        topo=args.topo,
        tc_file=getattr(args, 'tc_file', None),
        radius=args.radius,
        interp_method=args.interp_method,
        parallel=args.parallel,
        chunk_size=args.chunk_size,
        atm=args.atm,
        window_mode=args.window_mode,
        decimate=args.decimate,
        decimate_threshold=args.decimate_threshold
    )
    gravity_result = gravity_reduction.run(['faye'])
    Dg_file = gravity_result['output_files'][1]  # Gridded Faye anomalies
    Dg_ds = xr.open_dataset(Dg_file)
    Dg = Dg_ds['Dg']
    print(f'Computed Faye anomalies (Dg) saved to {Dg_file}')
    
    
    # Step 2: Calculate reference gravity anomalies from GGM (Dg_ggm)
    ggm_synthesis = GGMSynthesis(
        model=args.model,
        max_deg=args.max_deg,
        model_dir=model_dir,
        output_dir=output_dir,
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        parallel=args.parallel,
        tide_system=args.gravity_tide,
        converted=True,
        bbox=bbox,
        bbox_offset=args.bbox_offset,
        grid_size=grid_size,
        grid_unit=grid_unit,
        proj_name=args.proj_name,
        icgem=args.icgem
    )
    ggm_result = ggm_synthesis.run(['gravity-anomaly'])
    Dg_ggm_file = ggm_result['output_files'][0]
    Dg_ggm_ds = xr.open_dataset(Dg_ggm_file)
    Dg_ggm = Dg_ggm_ds['Dg']
    print(f'Computed reference gravity anomalies (Dg_ggm) saved to {Dg_ggm_file}')
    
    
    # Step 3: Compute residual gravity anomalies (D_res = Dg - Dg_ggm)
    if not (Dg_ds['lon'].equals(Dg_ggm_ds['lon']) and Dg_ds['lat'].equals(Dg_ggm_ds['lat'])):
        raise ValueError('Grids for Dg and Dg_ggm do not match')
    print('Estimating residual gravity anomalies as the difference between gridded surface and reference anomalies...')
    D_res = Dg - Dg_ggm
    D_res_ds = xr.Dataset({'D_res': D_res}, coords=Dg_ds.coords)
    print('Computed residual gravity anomalies.\n')
    
    # Step 4: Calculate the residual geoid (N_res)
    print('Computing the residual geoid as the sum of the inner and outer zones contributions...')
    sub_grid = bbox  # Use the same grid as input for simplicity
    residual_geoid = ResidualGeoid(
        res_anomaly=D_res_ds.rename({'D_res': 'Dg'}),
        sph_cap=args.sph_cap,
        sub_grid=sub_grid,
        method=args.method,
        ellipsoid=args.ellipsoid,
        nmax=args.max_deg,
        window_mode=args.window_mode
    )
    N_res = residual_geoid.compute_geoid()
    print(f'Saving residual geoid to {output_dir}/N_res.nc')
    save_to_netcdf(
        data=N_res,
        lon=residual_geoid.res_anomaly_P['lon'].values,
        lat=residual_geoid.res_anomaly_P['lat'].values,
        dataset_key='N_res',
        filepath=output_dir / 'N_res.nc',
        tide_system=ggm_tide,
        method=METHODS_DICT[args.method],
    )

    print('Residual geoid computation completed.\n')
    
    # Step 5: Calculate reference geoid (N_ggm)
    ggm_synthesis = GGMSynthesis(
        model=args.model,
        max_deg=args.max_deg,
        model_dir=model_dir,
        output_dir=output_dir,
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        parallel=args.parallel,
        tide_system=args.gravity_tide,
        converted=True,
        bbox=bbox,
        grid_size=grid_size,
        grid_unit=grid_unit,
        proj_name=args.proj_name,
        icgem=args.icgem
    )

    ggm_result = ggm_synthesis.run(['reference-geoid'])
    N_ggm_file = ggm_result['output_files'][0]
    N_ggm_ds = xr.open_dataset(N_ggm_file)
    N_ggm = N_ggm_ds['N_ref']
    
    print(f'Computed reference geoid saved to {N_ggm_file}')
    
    
    # Step 6: Calculate indirect effect (N_ind)
    topo_quantities = TopographicQuantities(
        topo=args.topo,
        ref_topo=args.ref_topo,
        model_dir=model_dir,
        output_dir=output_dir,
        ellipsoid=args.ellipsoid,
        chunk_size=args.chunk_size,
        radius=args.radius,
        parallel=args.parallel,
        bbox=bbox,
        bbox_offset=args.bbox_offset,
        grid_size=args.ind_grid_size,
        proj_name=args.proj_name,
        window_mode=args.window_mode,
        interp_method=args.interp_method,
    )
    topo_result = topo_quantities.run(['indirect-effect'])
    N_ind_file = topo_result['output_files'][0]
    N_ind_ds = xr.open_dataset(N_ind_file)
    N_ind = N_ind_ds['N_ind']
    print(f'Computed indirect effect saved to {N_ind_file}')
    
    
    # Step 7: Calculate total geoid (N = N_ggm + N_res + N_ind)
    if args.ind_grid_size != args.grid_size:
        print('Resampling indirect effect to the same grid as the reference and residual geoids...')
        N_ind = N_ind.interp(lon=N_ggm_ds['lon'], lat=N_ggm_ds['lat'], method='linear')
    
    
    print('Calculating total geoid as the sum of reference, residual, and indirect effects...')
    N = N_ggm.values + N_res + N_ind.values
    
    output_file = output_dir / 'N.nc'
    
    
    # Convert tide system if needed
    
    
    if args.target_tide_system != ggm_tide:
        from geoidlab.tide import GeoidTideSystemConverter
        import numpy as np
        print(f'Converting geoid from {ggm_tide} to {args.target_tide_system} tide system...')
        phi, _ = np.meshgrid(N_ggm_ds['lat'], N_ggm_ds['lon'])
        # phi = phi.flatten()
        converter = GeoidTideSystemConverter(phi=phi, geoid=N)
        conversion_map = {
            ('mean_tide', 'tide_free'): 'mean2free',
            ('tide_free', 'mean_tide'): 'free2mean',
            ('mean_tide', 'zero_tide'): 'mean2zero',
            ('zero_tide', 'mean_tide'): 'zero2mean',
            ('zero_tide', 'tide_free'): 'zero2free',
            ('tide_free', 'zero_tide'): 'free2zero'
        }
        conversion_key = (ggm_tide, args.target_tide_system)
        if conversion_key not in conversion_map:
            raise ValueError(f'No conversion defined from {ggm_tide} to {args.target_tide_system}')
        
        # Perform conversion
        conversion_method = getattr(converter, conversion_map[conversion_key])
        N = conversion_method()
        print(f'Geoid converted to {args.target_tide_system} tide system.')
    
    print(f'Writing geoid to {output_file}')
    save_to_netcdf(
        data=N,
        lon=N_ggm['lon'].values,
        lat=N_ggm['lat'].values,
        dataset_key='N',
        filepath=output_file,
        tide_system=args.target_tide_system if args.target_tide_system else ggm_tide,
        method=METHODS_DICT[args.method],
    )

    print(f'Geoid heights written to {output_file}.\n')
    
    print('Geoid computation completed successfully.\n\n\n')
    
    return 0
    
if __name__ == '__main__':
    sys.exit(main())