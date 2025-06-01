# Basic Usage Examples

This section provides basic examples of using GeoidLab. For more detailed examples, check out our Jupyter notebooks in the [tutorial repository](https://github.com/cikelly/geoidlab-tutorial).

## Example 1: Computing Helmert Anomalies

```python
from geoidlab.cli.commands.helmert import GravityReduction

# Initialize the reduction
reducer = GravityReduction(
    input_file="gravity.csv",
    model="EGM2008",
    grid=True,
    grid_size=1,
    grid_unit="minutes",
    bbox=[-5, 5, -5, 5],
    grid_method="kriging"
)

# Run the reduction
result = reducer.run(['helmert'])
```

## Example 2: Computing Residual Geoid

```python
from geoidlab.geoid import ResidualGeoid
import xarray as xr

# Load residual anomalies
anomalies = xr.open_dataset('residual_anomalies.nc')

# Initialize residual geoid computation
residual = ResidualGeoid(
    res_anomaly=anomalies,
    sph_cap=1.0,
    method='hg',
    nmax=2190,
    window_mode='cap'
)

# Compute residual geoid
N_res = residual.compute_geoid()
```

## Example 3: Tide System Conversion

```python
from geoidlab.tide import GeoidTideSystemConverter
import numpy as np

# Initialize converter
converter = GeoidTideSystemConverter(
    phi=lat_grid,
    geoid=geoid_heights
)

# Convert from mean tide to tide free system
geoid_tide_free = converter.mean2free()
```

For more examples and detailed explanations, see the individual example pages in this section.
