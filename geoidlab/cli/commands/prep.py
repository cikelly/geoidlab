############################################################
# Preprocessing CLI scaffold                               #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################

import argparse
from pathlib import Path

import pandas as pd


PLATFORMS = ['terrestrial', 'airborne', 'marine']
DATA_TYPES = ['gravity', 'free_air_anomaly']


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column aliases to GeoidLab's canonical names."""
    column_mapping = {
        'lon': ['lon', 'long', 'longitude', 'x'],
        'lat': ['lat', 'lati', 'latitude', 'y'],
        'height': ['height', 'h', 'elevation', 'elev', 'z'],
        'gravity': ['gravity', 'g', 'grav', 'acceleration'],
        'Dg': ['dg', 'anomaly', 'gravity_anomaly', 'free_air_anomaly', 'faa'],
    }

    normalized = {}
    for column in df.columns:
        lower = column.lower()
        normalized[column] = next(
            (canonical for canonical, aliases in column_mapping.items() if lower in aliases),
            column
        )

    return df.rename(columns=normalized)


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

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(input_path)
    elif input_path.suffix.lower() == '.txt':
        df = pd.read_csv(input_path, delimiter='\t')
    else:
        raise ValueError(f'Unsupported file format: {input_path.suffix}')

    df = _normalize_columns(df)

    required_columns = ['lon', 'lat', 'height']
    if args.data_type == 'gravity':
        required_columns.append('gravity')
    else:
        required_columns.append('Dg')

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f'Input file is missing required columns for platform={args.platform} and data_type={args.data_type}: '
            f'{missing}'
        )

    normalized = df[required_columns].copy()
    normalized.attrs = {
        'platform': args.platform,
        'data_type': args.data_type,
        'tide_system': args.tide_system,
    }

    if args.tide_system is not None:
        normalized['tide_system'] = args.tide_system
    normalized['platform'] = args.platform
    normalized['data_type'] = args.data_type

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    print(f'Normalized data written to {output_path}')
