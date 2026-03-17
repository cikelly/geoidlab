############################################################
# Preprocessing CLI scaffold                               #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################

import argparse
from geoidlab.preprocessing import (
    DATA_TYPES,
    PLATFORMS,
    write_standardized_observations,
)


def add_prep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Input survey file to normalize into a GeoidLab-ready table.')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output CSV path for the normalized table.')
    parser.add_argument('--platform', type=str, required=True, choices=PLATFORMS,
                        help='Observation platform represented by the input file.')
    parser.add_argument('--data-type', type=str, required=True, choices=DATA_TYPES,
                        help='Quantity stored in the input file.')
    parser.add_argument('--tide-system', type=str, default=None,
                        choices=['mean_tide', 'tide_free', 'zero_tide'],
                        help='Tide system metadata to store in the normalized file.')


def main(args=None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            description=(
                'Normalize survey data into the canonical tables consumed by GeoidLab. '
                'This is a scaffold for future terrestrial, airborne, and marine preprocessing workflows.'
            )
        )
        add_prep_arguments(parser)
        args = parser.parse_args()

    output_path = write_standardized_observations(
        data=args.input_file,
        output_file=args.output_file,
        platform=args.platform,
        data_type=args.data_type,
        tide_system=args.tide_system,
    )
    print(f'Normalized data written to {output_path}')
