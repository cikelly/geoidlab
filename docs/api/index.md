# Geoid Computation API

This section provides detailed API documentation for GeoidLab's core modules.

## Core Modules

### Geoid Module (`geoidlab.geoid`)

The geoid module provides core functionality for geoid modeling and computation:

#### ResidualGeoid Class

Main class for computing residual geoid heights using various kernel modifications:

```{eval-rst}
.. py:class:: ResidualGeoid

   Compute residual geoid heights using the Stokes' integral.

   .. py:method:: __init__(res_anomaly, sph_cap=1.0, sub_grid=None, method='hg', ellipsoid='wgs84', nmax=None, window_mode='cap')
      
      :param res_anomaly: Gridded residual gravity anomalies as xarray Dataset
      :param sph_cap: Spherical cap radius for integration (degrees)
      :param sub_grid: Subgrid extents (min_lon, max_lon, min_lat, max_lat)
      :param method: Kernel modification method ('hg', 'wg', 'og', or 'ml')
      :param ellipsoid: Reference ellipsoid ('wgs84' or 'grs80') 
      :param nmax: Maximum degree for modified kernels
      :param window_mode: Integration window mode ('cap' or 'fixed')
```

### Gravity Module (`geoidlab.gravity`)

Core functions for gravity field computations:

```{eval-rst}
.. py:function:: normal_gravity(phi, ellipsoid='wgs84')
   
   Compute normal gravity using truncated series expansion.

   :param phi: Geodetic latitude (degrees)
   :param ellipsoid: Reference ellipsoid ('wgs84' or 'grs80')
   :returns: Normal gravity (m/s²)

.. py:function:: normal_gravity_somigliana(phi, ellipsoid='wgs84') 
   
   Compute normal gravity using Somigliana's formula.

   :param phi: Geodetic latitude (degrees)
   :param ellipsoid: Reference ellipsoid ('wgs84' or 'grs80')
   :returns: Normal gravity (m/s²)

.. py:function:: gravity_anomalies(lat, gravity, elevation, ellipsoid='wgs84', atm=False, atm_method='noaa')
   
   Compute free-air and Bouguer gravity anomalies.
   
   :param lat: Latitude array (degrees)
   :param gravity: Surface gravity array (mGal)
   :param elevation: Station elevations (m)
   :param ellipsoid: Reference ellipsoid
   :param atm: Apply atmospheric correction
   :param atm_method: Atmospheric correction method ('noaa')
   :returns: Tuple of (free_air_anomaly, bouguer_anomaly) in mGal
```

### Global Geopotential Models (`geoidlab.ggm`)

Classes for handling spherical harmonic gravity field models:

```{eval-rst}
.. py:class:: GlobalGeopotentialModel
   
   Primary class for synthesizing gravity field functionals from spherical harmonic coefficients.

   .. py:method:: __init__(shc=None, model_name=None, ellipsoid='wgs84', nmax=90, grav_data=None, zonal_harmonics=True, model_dir='downloads', chunk_size=None, dtm_model=None)

   .. py:method:: gravity_anomaly(parallel=True)
      
      Synthesize gravity anomalies.
      
      :param parallel: Enable parallel processing
      :returns: Array of gravity anomalies (mGal)

   .. py:method:: geoid(T=None, icgem=False, parallel=False)
      
      Compute geoid heights.
      
      :param T: Precomputed disturbing potential (optional)
      :param icgem: Apply topographic effect using DTM2006
      :returns: Array of geoid heights (m)
```

### Constants (`geoidlab.constants`)

Reference systems and physical constants:

```{eval-rst}
.. py:function:: grs80()
   
   Return GRS80 ellipsoid parameters.
   
   :returns: Dictionary of GRS80 constants

.. py:function:: wgs84()
   
   Return WGS84 ellipsoid parameters.
   
   :returns: Dictionary of WGS84 constants

.. py:function:: earth()
   
   Return Earth/Geoid constants including W0.
   
   :returns: Dictionary of Earth constants
```

## Data Handling

### Digital Elevation Models (`geoidlab.dem`)

Functions and utilities for handling digital elevation model data:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   geoidlab.dem
```

### Digital Terrain Models (`geoidlab.dtm`)

Tools for computing terrain effects:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   geoidlab.dtm
```

### ICGEM Interface (`geoidlab.icgem`)

Functions for handling ICGEM (International Centre for Global Earth Models) data:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   geoidlab.icgem

.. py:function:: get_ggm_tide_system(icgem_file, model_dir)
   
   Get the tide system of a global geopotential model.
   
   :param icgem_file: Path to ICGEM file
   :param model_dir: Directory containing the model
   :returns: Tide system string ('mean_tide', 'zero_tide', or 'tide_free')
```

## Mathematical Tools

### Legendre Functions (`geoidlab.legendre`)

Classes for computing fully-normalized associated Legendre functions:

```{eval-rst}
.. py:class:: ALF

   Base class for associated Legendre functions.

.. py:class:: ALFsGravityAnomaly

   Specialized class for gravity anomaly computations.
```

### Stokes' Kernel (`geoidlab.stokes_func`)

Implementations of the Stokes' kernel and its modifications:

```{eval-rst}
.. py:class:: Stokes4ResidualGeoid

   Class implementing Stokes' kernel modifications for residual geoid determination.
   Includes Heck & Gruninger (hg), Wong & Gore (wg), original Stokes (og), and Meissl (ml) kernels.
```

## Coordinate Systems and Transformations

### Coordinate Transformations (`geoidlab.coordinates`)

Functions for coordinate system conversions:

```{eval-rst}
.. py:function:: geodetic2geocentric(lat, height, ellipsoid='wgs84')
   
   Convert geodetic coordinates to geocentric.
   
   :param lat: Geodetic latitude (degrees)
   :param height: Ellipsoidal height (meters)
   :param ellipsoid: Reference ellipsoid
   :returns: Geocentric coordinates
```

### Terrain Tools (`geoidlab.terrain`)

Utilities for terrain corrections and topographic reductions.

### Tide Systems (`geoidlab.tide`) 

Classes for tide system conversions:

```{eval-rst}
.. py:class:: GravityTideSystemConverter

   Convert gravity data between different tide systems.

   .. py:method:: __init__(path_to_data=None, data=None, k=0.3, h=0.6, d=1.53)
      
      :param path_to_data: Path to data file
      :param data: Data as array, DataFrame or dict
      :param k: Elastic Love number
      :param h: Elastic Love number
      :param d: Dynamic form factor

   .. py:method:: mean2free()
      Convert from mean tide to tide-free system
   
   .. py:method:: free2mean()
      Convert from tide-free to mean tide system
   
   .. py:method:: mean2zero()
      Convert from mean tide to zero tide system
   
   .. py:method:: zero2mean() 
      Convert from zero tide to mean tide system
```

## Utilities and Tools

### General Utilities (`geoidlab.utils`)

Helper functions and I/O utilities:

```{eval-rst}
.. py:data:: DATASETS_CONFIG
   
   Configuration for common dataset variables including units and descriptions.

   Supported variables include:
   - Gravity anomalies (free-air, Bouguer, Helmert)
   - Geoid heights (residual, reference, total)
   - Height anomalies
   - Terrain corrections
```

### Mapping Tools (`geoidlab.mapping`)

Tools for data visualization and mapping.
