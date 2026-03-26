# Usage Guide

## Command Line Interface

GeoidLab provides a comprehensive command-line interface (CLI) for geoid computation. The basic command structure is:

```bash
geoidlab <command> [options]
```

## Available Commands

- `prep`: Normalize survey data into GeoidLab-ready tables
- `ggm`: Synthesize gravity field functionals from a global geopotential model
- `geoid`: Complete geoid computation using RCR method
- `reduce`: Perform gravity reductions
- `topo`: Compute topographic quantities
- `viz`: Visualize results
- `ncinfo`: Inspect NetCDF outputs

## Basic Workflow

1. Observation preparation
2. Gravity Reductions
3. Terrain and reference-field modelling
4. Residual computation
5. Geoid Determination
6. Final Model Generation

Most users either run the workflow from the CLI directly or start from the template configuration file in `docs/geoidlab.cfg`.
