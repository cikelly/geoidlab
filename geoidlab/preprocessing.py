############################################################
# Preprocessing utilities for canonical gravity inputs     #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################

from __future__ import annotations

from pathlib import Path

import pandas as pd


PLATFORMS = ['terrestrial', 'airborne', 'marine']
DATA_TYPES = ['gravity', 'free_air_anomaly']

CANONICAL_SCHEMAS = {
    ('terrestrial', 'gravity'): ['lon', 'lat', 'height', 'gravity'],
    ('airborne', 'gravity'): ['lon', 'lat', 'height', 'gravity'],
    ('marine', 'free_air_anomaly'): ['lon', 'lat', 'height', 'Dg'],
}

COLUMN_MAPPING = {
    'lon': ['lon', 'long', 'longitude', 'x'],
    'lat': ['lat', 'lati', 'latitude', 'y'],
    'height': ['height', 'h', 'elevation', 'elev', 'z'],
    'gravity': ['gravity', 'g', 'grav', 'acceleration'],
    'Dg': ['dg', 'anomaly', 'gravity_anomaly', 'free_air_anomaly', 'faa'],
}


def read_observations(input_file: str | Path) -> pd.DataFrame:
    """Read a gravity-survey table from a supported tabular format."""
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(input_path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(input_path)
    if suffix == '.txt':
        return pd.read_csv(input_path, delimiter='\t')

    raise ValueError(f'Unsupported file format: {input_path.suffix}')


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common aliases to GeoidLab's canonical input column names."""
    normalized = {}
    for column in df.columns:
        lower = column.lower()
        normalized[column] = next(
            (canonical for canonical, aliases in COLUMN_MAPPING.items() if lower in aliases),
            column
        )

    return df.rename(columns=normalized)


def canonical_schema(platform: str, data_type: str) -> list[str]:
    """Return the canonical column schema for a supported platform/data pair."""
    key = (platform, data_type)
    if key not in CANONICAL_SCHEMAS:
        raise ValueError(
            f'Unsupported platform/data_type combination: platform={platform!r}, data_type={data_type!r}.'
        )
    return CANONICAL_SCHEMAS[key]


def standardize_observations(
    data: pd.DataFrame | str | Path,
    platform: str,
    data_type: str,
    tide_system: str | None = None,
) -> pd.DataFrame:
    """Normalize an observation table into a GeoidLab canonical schema."""
    if platform not in PLATFORMS:
        raise ValueError(f'Unsupported platform={platform!r}. Expected one of {PLATFORMS}.')
    if data_type not in DATA_TYPES:
        raise ValueError(f'Unsupported data_type={data_type!r}. Expected one of {DATA_TYPES}.')

    df = read_observations(data) if isinstance(data, (str, Path)) else data.copy()
    df = normalize_columns(df)

    required_columns = canonical_schema(platform, data_type)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f'Input observations are missing required columns for platform={platform} and data_type={data_type}: '
            f'{missing}'
        )

    standardized = df[required_columns].copy()
    if tide_system is not None:
        standardized['tide_system'] = tide_system
    standardized['platform'] = platform
    standardized['data_type'] = data_type
    return standardized


def write_standardized_observations(
    data: pd.DataFrame | str | Path,
    output_file: str | Path,
    platform: str,
    data_type: str,
    tide_system: str | None = None,
) -> Path:
    """Standardize a table and write it to a CSV file for GeoidLab ingestion."""
    standardized = standardize_observations(
        data=data,
        platform=platform,
        data_type=data_type,
        tide_system=tide_system,
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)
    return output_path
