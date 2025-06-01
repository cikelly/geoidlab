# Introduction

GeoidLab is a Python package designed for geoid computation using the remove-compute-restore (RCR) method. It provides a comprehensive command-line interface for automated geoid computation, handling everything from data preparation to final geoid model generation.

## Core Features

- Complete RCR implementation for geoid determination
- Multiple Stokes' kernel modifications
- Automated data downloads (DEMs, GGMs)
- Advanced gridding and interpolation options
- Comprehensive tide system support
- Multiple reference ellipsoid support

## Supported Methods

GeoidLab implements several modifications of Stokes' kernel:

- Heck & Gruninger's modification
- Wong & Gore's modification
- Original Stokes' function
- Meissl's modification

## Data Handling

The package supports various data sources:

- Terrain data:
  - SRTM30PLUS
  - Copernicus DEM
  - NASADEM
  - GEBCO

- Global Geopotential Models from ICGEM
- Marine gravity data integration
- Multiple tide systems with automatic conversions
