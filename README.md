# GeoidLab

[![Language](https://img.shields.io/badge/python-3.10--3.12-blue.svg?style=flat-square)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/geoidlab.svg?style=flat-square)](https://pypi.org/project/geoidlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/geoidlab?style=flat-square&label=downloads)](https://pypi.org/project/geoidlab/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/cikelly/geoidlab/tests.yml?style=flat-square&label=tests)](https://github.com/cikelly/geoidlab/actions)
[![Documentation Status](https://readthedocs.org/projects/geoidlab/badge/?version=latest&style=flat-square)](https://geoidlab.readthedocs.io/en/latest/?badge=latest)

---
<div align="center">
  <img src="docs/logo/logo.png" alt="Logo" width="100"/>
</div>

<p></p>

**`GeoidLab`: A Modular and Automated Python Package for Geoid Computation.**

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Command-Line Interface](#command-line-interface)
- [Workflows](#workflows)
- [Documentation and Examples](#documentation-and-examples)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction
`GeoidLab` is a Python package for gravity-field analysis and geoid determination, centered on the remove-compute-restore (RCR) workflow. It includes command-line tools for preparing observation tables, synthesizing gravity-field quantities from global geopotential models, computing terrain quantities and gravity reductions, estimating geoid models, and visualizing NetCDF outputs.

## Features

### Core Functionality
- **End-to-end RCR workflow**
  - Gravity reduction
  - Terrain correction and indirect effects
  - Synthesis of gravity field functionals from global geopotential models
  - Residual anomaly computation
  - Geoid estimation with multiple modified Stokes kernels
  - Restoration of reference and terrain-related terms

- **Observation preprocessing**
  - Normalize terrestrial, airborne, and marine survey tables into GeoidLab-ready schemas
  - Accept CSV, Excel, and tab-delimited text inputs
  - Preserve tide-system metadata for downstream processing

- **Topographic modelling**
  - Built-in DEM support for `srtm30plus`, `srtm`, `cop`, `nasadem`, and `gebco`
  - Support for local DEM files, remote GDAL-readable DEM URLs, and cloud-optimized GeoTIFF sources
  - Terrain correction, indirect effect, secondary indirect effect, RTM anomaly, and RTM height anomaly
  - Reference topography synthesis from DTM2006.0 or Earth2014 SHC models
  - Optional variable-density workflows

- **Global geopotential model synthesis**
  - Automatic download of GGMs from ICGEM
  - Gravity anomaly
  - Gravity disturbance
  - Disturbing potential
  - Disturbing-potential derivative
  - Second radial derivative
  - Reference geoid
  - Height anomaly
  - Geoid-quasigeoid separation
  - Zero-degree term correction
  - Ellipsoidal correction

- **Geoid computation**
  - Modified Stokes kernels: `hg`, `wg`, `ml`, `og`
  - Station-based or gridded residual workflows
  - Tide-system handling for gravity, marine anomalies, and final geoid outputs

- **Visualization**
  - Publication-ready plotting of gridded NetCDF results
  - Support for `matplotlib` colormaps and GMT `.cpt` colormaps
  - Scalebar and contour options
  - Automatic plot generation for saved `geoid` outputs

### Reference Systems
- Built-in support for `wgs84` and `grs80`
- Optional custom ellipsoid definitions through CLI/config inputs

## Installation
GeoidLab currently targets Python `>=3.10,<3.13`.

### Install from PyPI
```bash
pip install geoidlab
```

### Install from source
```bash
git clone https://github.com/cikelly/geoidlab.git
cd geoidlab
pip install -e .
```

### Development install
```bash
git clone https://github.com/cikelly/geoidlab.git
cd geoidlab
pip install -e .[dev]
```

## Command-Line Interface
GeoidLab exposes a single CLI entry point:

```bash
geoidlab <subcommand> [options]
```

Current subcommands:
- `prep`   Normalize input survey tables into GeoidLab-ready CSV files
- `ggm`    Synthesize gravity field functionals from a global geopotential model
- `topo`   Compute topographic quantities from a DEM
- `reduce` Compute gravity reductions
- `geoid`  Run the full RCR geoid workflow
- `viz`    Visualize NetCDF outputs
- `ncinfo` Inspect a NetCDF file

Examples:

```bash
geoidlab -h
geoidlab prep -h
geoidlab ggm -h
geoidlab geoid -h
geoidlab viz --help
```

## Workflows

### 1. Prepare observation tables
Use `prep` when raw field tables need to be normalized into the column schemas expected by GeoidLab.

```bash
geoidlab prep \
  --input-file raw_gravity.xlsx \
  --output-file surface_gravity.csv \
  --platform terrestrial \
  --data-type gravity \
  --tide-system mean_tide
```

Supported platform and data-type combinations currently include:
- terrestrial gravity
- airborne gravity
- marine free-air anomaly

### 2. Run geoid computation with a config file
The simplest way to run a full workflow is through a configuration file.

1. Create a working directory and place your observation files there.
2. Copy the default template config:

```bash
mkdir Brazil && cd Brazil
geoidlab --init
```

3. Edit `geoidlab.cfg`. In the `[subcommand]` section, set:

```ini
[subcommand]
command = geoid
```

4. Run the workflow:

```bash
geoidlab -c geoidlab.cfg
```

`geoidlab --config geoidlab.cfg` does the same thing. For backward compatibility, bare `geoidlab -c` still initializes the template config, but `geoidlab --init` is the clearer form.

By default, results are written beneath `GeoidProject/`, with downloads in `GeoidProject/downloads` and outputs in `GeoidProject/results`.

### 3. Useful CLI parameters
Common options vary by subcommand, but these are among the most important:

- `--method`: Stokes kernel modification for `geoid` (`hg`, `wg`, `ml`, `og`)
- `--grid-method`: Gridding method for reductions (`linear`, `spline`, `kriging`, `rbf`, `idw`, `biharmonic`, `gpr`, `lsc`)
- `--gravity-tide`: Tide system of gravity observations
- `--marine-tide-system`: Tide system of marine anomaly observations
- `--target-tide-system`: Output tide system for geoid results
- `--ellipsoid`: `wgs84`, `grs80`, or a custom ellipsoid JSON string
- `--variable-density`: Enable variable-density workflows where supported

## Documentation and Examples
- Documentation source lives in [`docs/`](docs/)
- Usage and examples are in [`docs/usage/`](docs/usage/) and [`docs/examples/`](docs/examples/)
- The default configuration template is [`docs/geoidlab.cfg`](docs/geoidlab.cfg)
- Additional walkthrough material is available in the [tutorial repository](https://github.com/cikelly/geoidlab-tutorial)

The `Notebooks/` directory currently contains sample inputs, figures, and generated outputs used in examples and development work.

## Contributing
Contributions are welcome.

1. Fork the repository.
2. Clone your fork and install the package in editable mode.
3. Create a branch for your change.
4. Add or update tests where appropriate.
5. Submit a pull request.

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

## References
- Kelly, C.I., V.G. Ferreira, D. Yang, S.A. Andam-Akorful, D. Yan1, S. Osah, C.M. Hancock, G. Jing, A.T. Kabo-bah, (2026): GeoidLab: An Automated Open-Source Python Toolbox for Gravity-Field Analysis and Vertical Datum Unification (Under review)

- Yakubu, C. I., Ferreira, V. G. and Asante, C. Y., (2017): [Towards the Selection of an Optimal Global Geopotential Model for the Computation of the Long-Wavelength Contribution: A Case Study of Ghana, Geosciences, 7(4), 113](http://www.mdpi.com/2076-3263/7/4/113)

- C. I. Kelly, S. A. Andam-Akorful, C. M. Hancock, P. B. Laari & J. Ayer (2021): [Global gravity models and the Ghanaian vertical datum: challenges of a proper definition, Survey Review, 53(376), 44-54](https://doi.org/10.1080/00396265.2019.1684006)
