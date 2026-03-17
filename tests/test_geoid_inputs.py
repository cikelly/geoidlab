import argparse

import pandas as pd
import pytest

from geoidlab.cli.commands.geoid import ResidualAnomalyComputation
from geoidlab.cli.commands.prep import main as prep_main


def test_marine_requires_station_mode(tmp_path) -> None:
    input_file = tmp_path / "gravity.csv"
    marine_file = tmp_path / "marine.csv"

    pd.DataFrame(
        {"lon": [0.0], "lat": [0.0], "gravity": [980000.0], "height": [10.0]}
    ).to_csv(input_file, index=False)
    pd.DataFrame(
        {"lon": [0.0], "lat": [0.0], "height": [0.0], "Dg": [5.0]}
    ).to_csv(marine_file, index=False)

    computation = ResidualAnomalyComputation(
        input_file=str(input_file),
        marine_data=str(marine_file),
        marine_data_type="free_air_anomaly",
        marine_tide_system="mean_tide",
        model="EGM2008",
        gravity_tide="mean_tide",
        residual_method="gridded",
        proj_name=str(tmp_path / "proj"),
    )

    with pytest.raises(ValueError, match='Marine gravity is currently supported only with residual_method="station"'):
        computation._load_marine_data()


def test_prep_scaffold_normalizes_columns(tmp_path) -> None:
    input_file = tmp_path / "marine_input.csv"
    output_file = tmp_path / "normalized.csv"

    pd.DataFrame(
        {
            "longitude": [1.0],
            "latitude": [2.0],
            "elev": [3.0],
            "faa": [4.0],
        }
    ).to_csv(input_file, index=False)

    args = argparse.Namespace(
        input_file=str(input_file),
        output_file=str(output_file),
        platform="marine",
        data_type="free_air_anomaly",
        tide_system="mean_tide",
    )
    prep_main(args)

    normalized = pd.read_csv(output_file)
    assert list(normalized.columns) == ["lon", "lat", "height", "Dg", "tide_system", "platform", "data_type"]
    assert normalized.loc[0, "Dg"] == 4.0
    assert normalized.loc[0, "platform"] == "marine"
