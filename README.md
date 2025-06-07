# GeoidLab

[![Language](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/geoidlab.svg?style=flat-square)](https://pypi.org/project/geoidlab/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/cikelly/geoidlab/tests.yml?style=flat-square&label=tests)](https://github.com/cikelly/geoidlab/actions)
[![Coverage](https://img.shields.io/codecov/c/github/cikelly/geoidlab?style=flat-square)](https://codecov.io/gh/cikelly/geoidlab)
[![Documentation Status](https://readthedocs.org/projects/geoidlab/badge/?version=latest&style=flat-square)](https://geoidlab.readthedocs.io/en/latest/?badge=latest)
[![GitHub Release](https://img.shields.io/github/v/release/cikelly/geoidlab?color=yellow&label=version&style=flat-square)](https://github.com/cikelly/geoidlab/releases)

---
<div align="center">
  <img src="docs/logo/logo.png" alt="Logo" width="200"/>
</div>

<p></p>

**`GeoidLab`: A Modular and Automated Python Package for Geoid Computation.**



## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Command-Line Interface](#command-line-interface)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction
`GeoidLab` is a Python package that implements the remove-compute-restore (RCR) method for geoid determination. It provides a comprehensive command-line interface for automated geoid computation, handling everything from data preparation to final geoid model generation.

## Features

### Core Functionality
- **Complete RCR Implementation**: Handles all steps of the remove-compute-restore method:
  - Data preparation (free-air anomalies, terrain corrections)
  - Residual computation and gridding
  - Geoid determination with multiple kernel options
  - Final model restoration

### Supported Methods
- **Multiple Stokes' Kernel Modifications**:
  - Heck & Gruninger (hg)
  - Wong & Gore (wg)
  - Original Stokes' (og)
  - Meissl (ml)

### Data Handling
- **Automated Data Acquisition**:
  - Terrain data from multiple sources (SRTM30PLUS, Copernicus DEM, NASADEM, GEBCO)
  - Global Geopotential Models from ICGEM
  - Marine gravity data integration
  
- **Tide System Support**:
  - Mean tide
  - Zero tide
  - Tide-free
  - Automatic tide system conversions

### Computational Features
- **Advanced Gridding Options**:
  - Multiple interpolation methods (linear, spline, kriging, etc.)
  - Flexible grid size and unit specifications
  - Customizable computation windows

### Reference Systems
- Supports both WGS84 and GRS80 reference ellipsoids
- Handles various coordinate transformations and corrections

## Installation
`GeoidLab` can be installed using `conda`, `mamba`, or `pip`.

### Using Conda/Mamba
```bash
conda create -n geoid_env -y
mamba install -c conda-forge geoidlab -y
```

### Using Pip
```bash
pip install geoidlab
```

### Troubleshooting
- Ensure you have Python 3.10 or higher installed
- For `conda`/`mamba`, make sure the `conda-forge` channel is enabled
- Install required dependencies listed in `requirements.txt`

## Command-Line Interface
`GeoidLab` provides a comprehensive CLI with several subcommands for geoid computation:

```bash
# Basic command structure
geoidlab <command> [options]

# Available commands
geoidlab geoid    # Complete geoid computation using RCR method
geoidlab reduce   # Perform gravity reductions
geoidlab topo     # Compute topographic quantities
geoidlab viz      # Visualize results
```

### Geoid Computation Workflow
The main geoid computation follows these stages:

1. **Data Preparation**
   ```bash
   # Process terrestrial gravity data
   geoidlab reduce --input gravity.csv --model EGM2008 \
                   --grid-size 1 --grid-unit minutes \
                   --bbox [-5,5,-5,5] --grid-method kriging
   ```

2. **Residual Computation**
   ```bash
   # Compute residual geoid using Heck & Gruninger modification
   geoidlab geoid --method hg --sph-cap 1.0 \
                  --max-deg 2190 --window-mode cap
   ```

3. **Final Geoid Model**
   ```bash
   # Generate final geoid with tide system conversion
   geoidlab geoid --target-tide-system tide_free \
                  --gravity-tide mean_tide
   ```

### Key Parameters
- `--method`: Kernel modification ['hg', 'wg', 'ml', 'og']
- `--grid-method`: Interpolation method ['linear', 'spline', 'kriging', 'rbf', 'idw']
- `--tide-system`: Tide system ['mean_tide', 'zero_tide', 'tide_free']
- `--ellipsoid`: Reference ellipsoid ['wgs84', 'grs80']

## Examples
Visit the [tutorial repository](https://github.com/cikelly/geoidlab-tutorial) for comprehensive examples, including:
- Complete workflow demonstrations
- Different kernel modification comparisons
- Data preparation guides
- Visualization examples

The package includes example Jupyter notebooks in the `Notebooks/` directory that demonstrate various aspects of geoid computation.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Clone your fork and set up a development environment:
   ```bash
   git clone https://github.com/your-username/geoidlab.git
   cd geoidlab
   pip install -e .
   ```
3. Create a new branch for your feature or bug fix.
4. Submit a pull request.

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.


See the [tutorial repo](https://github.com/cikelly/geoidlab-tutorial) for detailed examples of using `GeoidLab`.

## References
- Yakubu, C. I., Ferreira, V. G. and Asante, C. Y., (2017): [Towards the Selection of an Optimal Global Geopotential
Model for the Computation of the Long-Wavelength Contribution: A Case Study of Ghana, Geosciences, 7(4), 113](http://www.mdpi.com/2076-3263/7/4/113)

- C. I. Kelly, S. A. Andam-Akorful, C. M. Hancock, P. B. Laari & J. Ayer (2021): [Global gravity models and the Ghanaian vertical datum: challenges of a proper definition, Survey Review, 53(376), 44–54](https://doi.org/10.1080/00396265.2019.1684006)
