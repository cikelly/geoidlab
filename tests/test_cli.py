"""
Tests for CLI functionality
"""
import os
import pytest
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from geoidlab.cli.main import main, auto_visualize, copy_template_config
from geoidlab.cli.commands.topo import TopographicQuantities
from geoidlab.cli.utils.config_parser import parse_config_file

def test_auto_visualize(tmp_path) -> None:
    """Test that auto_visualize correctly processes NetCDF files"""
    # Create a mock results directory with some NetCDF files
    results_dir = tmp_path / "GeoidProject" / "results"
    results_dir.mkdir(parents=True)
    
    # Create dummy NetCDF files
    (results_dir / "test1.nc").write_text("")
    (results_dir / "test2.nc").write_text("")
    
    # Create mock args
    args = argparse.Namespace()
    args.proj_name = str(tmp_path / "GeoidProject")
    args.subcommand = "ggm"  # Not viz
    args.save = True
    
    # Mock the plot_main function
    called_files = []
    def mock_plot_main(args) -> None:
        called_files.append(args.filename)
    
    # Run auto_visualize with our mock
    import geoidlab.cli.main
    original_plot_main = geoidlab.cli.main.plot_main
    try:
        geoidlab.cli.main.plot_main = mock_plot_main
        auto_visualize(args)
    finally:
        geoidlab.cli.main.plot_main = original_plot_main
    
    # Verify both files were processed
    assert len(called_files) == 2
    # assert str(results_dir / "test1.nc") in called_files
    # assert str(results_dir / "test2.nc") in called_files
    assert [str(results_dir / "test1.nc")] in called_files
    assert [str(results_dir / "test2.nc")] in called_files

def test_auto_visualize_skip_conditions() -> None:
    """Test conditions where auto_visualize should skip processing"""
    args = argparse.Namespace()
    
    # Should skip if subcommand is viz
    args.subcommand = "viz"
    args.save = True
    auto_visualize(args)  # Should not raise any errors
    
    # Should skip if save is False
    args.subcommand = "ggm"
    args.save = False
    auto_visualize(args)  # Should not raise any errors
    
    # Should skip if no results directory
    args.save = True
    args.proj_name = "/nonexistent/path"
    auto_visualize(args)  # Should not raise any errors


def test_init_skips_existing_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "geoidlab.cfg"
    config_file.write_text("existing config")

    with pytest.raises(SystemExit) as excinfo:
        copy_template_config()

    assert excinfo.value.code == 0
    assert config_file.read_text() == "existing config"


def test_init_forced_replaces_existing_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "geoidlab.cfg"
    config_file.write_text("existing config")

    with pytest.raises(SystemExit) as excinfo:
        copy_template_config(forced=True)

    assert excinfo.value.code == 0
    assert config_file.read_text() != "existing config"
    assert "[subcommand]" in config_file.read_text()


def test_topo_config_accepts_local_dem_without_builtin_topo(tmp_path) -> None:
    """Config parsing should accept topo_file as the topo DEM source."""
    dem_file = tmp_path / "my_custom_dem_name.nc"
    dem_file.write_text("")
    config_file = tmp_path / "geoidlab.cfg"
    config_file.write_text(
        "\n".join(
            [
                "[subcommand]",
                "command = topo",
                "",
                "[grid]",
                "bbox = -1 1 -1 1",
                "grid_size = 30",
                "grid_unit = seconds",
                "",
                "[topography]",
                f"topo_file = {dem_file.name}",
                "topo_lon_name = lon",
                "topo_lat_name = lat",
                "topo_height_name = elevation",
            ]
        )
    )

    args = parse_config_file(
        str(config_file),
        argparse.Namespace(subcommand=None, func=None),
    )

    assert args.subcommand == "topo"
    assert args.topo is None
    assert args.topo_file == str(dem_file.resolve())
    assert args.topo_lon_name == "lon"
    assert args.topo_lat_name == "lat"
    assert args.topo_height_name == "elevation"


def test_topo_config_rejects_multiple_dem_sources(tmp_path) -> None:
    dem_file = tmp_path / "local.nc"
    dem_file.write_text("")
    config_file = tmp_path / "geoidlab.cfg"
    config_file.write_text(
        "\n".join(
            [
                "[subcommand]",
                "command = topo",
                "",
                "[grid]",
                "bbox = -1 1 -1 1",
                "",
                "[topography]",
                "topo = srtm",
                f"topo_file = {dem_file.name}",
            ]
        )
    )

    with pytest.raises(SystemExit):
        parse_config_file(
            str(config_file),
            argparse.Namespace(subcommand=None, func=None),
        )


def test_viz_config_accepts_plot_options_and_shared_project(tmp_path, capsys) -> None:
    nc_file = tmp_path / "result.nc"
    nc_file.write_text("")
    config_file = tmp_path / "geoidlab.cfg"
    config_file.write_text(
        "\n".join(
            [
                "[subcommand]",
                "command = viz",
                "",
                "[project]",
                "proj_name = CustomProject",
                "",
                "[viz]",
                f"filename = {nc_file.name}",
                "cmap = viridis",
                "share_cbar = true",
                "shared_cbar_orientation = horizontal",
                "shared_cbar_shrink = 0.75",
                "shared_cbar_pad = 0.04",
                "shared_cbar_font_size = 14",
                "contour = true",
                "contour_color = white",
                "contour_linewidth = 0.5",
                "contour_alpha = 0.6",
                "contour_levels = 10",
            ]
        )
    )

    args = parse_config_file(
        str(config_file),
        argparse.Namespace(subcommand=None, func=None),
    )

    captured = capsys.readouterr()
    assert "Ignoring unknown parameter" not in captured.out
    assert args.subcommand == "viz"
    assert args.proj_name == "CustomProject"
    assert args.filename == [str(nc_file.resolve())]
    assert args.cmap == ["viridis"]
    assert args.share_cbar is True
    assert args.shared_cbar_orientation == "horizontal"
    assert args.shared_cbar_shrink == 0.75
    assert args.shared_cbar_pad == 0.04
    assert args.shared_cbar_font_size == 14
    assert args.contour is True
    assert args.contour_color == "white"
    assert args.contour_linewidth == 0.5
    assert args.contour_alpha == 0.6
    assert args.contour_levels == "10"


def test_config_force_flag_is_parsed(tmp_path) -> None:
    config_file = tmp_path / "geoidlab.cfg"
    input_file = tmp_path / "gravity.csv"
    input_file.write_text("lon,lat,gravity,height\n0,0,980000,0\n")
    config_file.write_text(
        "\n".join(
            [
                "[subcommand]",
                "command = reduce",
                "",
                "[input_data]",
                f"input_file = {input_file.name}",
                "",
                "[ggm]",
                "model = EGM2008",
                "",
                "[grid]",
                "bbox = -1 1 -1 1",
                "",
                "[computation]",
                "force = True",
            ]
        )
    )

    args = parse_config_file(
        str(config_file),
        argparse.Namespace(subcommand=None, func=None),
    )

    assert args.force is True


def test_topo_indirect_effect_respects_force(tmp_path) -> None:
    output_file = tmp_path / "N_ind.nc"
    output_file.write_text("existing")

    class DummyTerrain:
        def __init__(self) -> None:
            self.calls = 0

        def indirect_effect(self, **kwargs) -> None:
            self.calls += 1

    workflow = TopographicQuantities.__new__(TopographicQuantities)
    workflow.output_dir = tmp_path
    workflow.parallel = False
    workflow.chunk_size = 1
    workflow.radius = 167.0
    workflow.ellipsoid = "wgs84"
    workflow.tq = DummyTerrain()
    workflow.force = False

    result = workflow.compute_ind()
    assert result["status"] == "skipped"
    assert workflow.tq.calls == 0

    workflow.force = True
    result = workflow.compute_ind()
    assert result["status"] == "success"
    assert workflow.tq.calls == 1
