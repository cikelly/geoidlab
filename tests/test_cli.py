"""
Tests for CLI functionality
"""
import os
import pytest
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from geoidlab.cli.main import main, auto_visualize

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
    assert str(results_dir / "test1.nc") in called_files
    assert str(results_dir / "test2.nc") in called_files

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
