import argparse
import sys

from geoidlab.cli.commands.reference import add_reference_arguments, main as ggm_main
from geoidlab.cli.commands.topo import add_topo_arguments, main as topo_main
from geoidlab.cli.commands.faye import add_faye_arguments, main as faye_main
from geoidlab.cli.commands.plot import add_plot_arguments, main as plot_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'GeoidLab: A toolkit for geodetic computations including gravity reductions, '
            'terrain quantities, GGM synthesis, geoid computation, and visualization.'
        ),
        epilog='Available commands: ggm, reduce, topo, viz, geoid'
    )
    parser.add_argument('--version', action='version', version='geoidlab 1.0.0')
    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands', required=False)
    
    # GGM subcommand
    ggm_parser = subparsers.add_parser('ggm', help='Synthesize gravity field functionals from a global geopotential model (GGM)')
    add_reference_arguments(ggm_parser)
    ggm_parser.set_defaults(func=ggm_main)
    
    # Topo
    topo_parser = subparsers.add_parser('topo', help='Compute topographic quantities from a Digital Elevation Model (DEM)')
    add_topo_arguments(topo_parser)
    topo_parser.set_defaults(func=topo_main)
    
    # Faye
    reduce_parser = subparsers.add_parser('reduce', help='Perform gravity reduction (Free-air, Bouguer, Faye/Helmert)')
    add_faye_arguments(reduce_parser)
    reduce_parser.set_defaults(func=faye_main)
    
    # Plot
    plot_parser = subparsers.add_parser('viz', help='Visualize data')
    add_plot_arguments(plot_parser)
    plot_parser.set_defaults(func=plot_main)
    
    args = parser.parse_args()
    return args.func(args)
    
    


if __name__ == '__main__':
    sys.exit(main())