# Basic Usage Examples

This section provides basic examples of using GeoidLab. For larger walkthroughs, see the [tutorial repository](https://github.com/cikelly/geoidlab-tutorial).

## Example 1: Compute Helmert anomalies from Python

```python
from geoidlab.cli.commands.reduce import GravityReduction

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

The equivalent CLI call is:

```bash
geoidlab reduce \
  --input-file gravity.csv \
  --model EGM2008 \
  --do helmert \
  --grid \
  --grid-size 1 \
  --grid-unit minutes \
  --bbox -5 5 -5 5 \
  --grid-method kriging
```

## Example 2: Compute a residual geoid from a gridded anomaly field

```python
from geoidlab.geoid import ResidualGeoid
import xarray as xr

# Load residual anomalies
anomalies = xr.open_dataset('residual_anomalies.nc')

# Initialize residual geoid computation
residual = ResidualGeoid(
    res_anomaly=anomalies,
    sub_grid=(-5, 5, -5, 5),
    sph_cap=1.0,
    method='hg',
    nmax=2190,
    window_mode='cap'
)

# Compute residual geoid heights
N_res = residual.compute_geoid()
```

## Example 3: Convert geoid values between tide systems

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

## Example 4: Normalize a survey table before running the workflow

```bash
geoidlab prep \
  --input-file raw_gravity.xlsx \
  --output-file surface_gravity.csv \
  --platform terrestrial \
  --data-type gravity \
  --tide-system mean_tide
```

For more examples and detailed explanations, see the companion pages in this documentation set and the tutorial repository.
