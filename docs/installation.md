# Installation

GeoidLab can be installed using pip, conda, or from source.

## Using pip

```bash
pip install geoidlab
```

## Using conda

```bash
conda install -c conda-forge geoidlab
```

## From Source

To install GeoidLab from source:

```bash
git clone https://github.com/cikelly/geoidlab.git
cd geoidlab
pip install -e .
```

## Dependencies

GeoidLab requires Python 3.10 or later. The main dependencies are:

- numpy
- scipy
- xarray
- pandas
- numba
- rasterio
- scikit-learn
- pyproj

These will be automatically installed when you install GeoidLab.
