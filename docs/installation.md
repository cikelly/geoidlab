# Installation

GeoidLab currently targets Python `>=3.10,<3.13`.

## Using pip

```bash
pip install geoidlab
```

## From source

```bash
git clone https://github.com/cikelly/geoidlab.git
cd geoidlab
pip install -e .
```

## Development install

```bash
git clone https://github.com/cikelly/geoidlab.git
cd geoidlab
pip install -e .[dev]
```

## Dependencies

GeoidLab installs its Python dependencies automatically. Core packages include:

- numpy
- scipy
- xarray
- pandas
- numba
- rasterio
- scikit-learn
- pyproj
