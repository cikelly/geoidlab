# GeoidLab

<div align="center">
  <img src="docs/logo/logo1.png" alt="Logo" width="200"/>
</div>

<p></p>
                                                                                  

**`GeoidLab` is a highly automated Python package developed for easy geoid modeling.**


The package comes packed with utilities for estimating a geoid model using Stokes' method, with options for:

- the original Stokes' kernel
- Wong and Gore's modification of Stokes' kernel
- Heck and Gruninger's modification of Stokes' kernel
- Terrain correction
- Residual Terrain Modeling (RTM)

`GeoidLab` uses the remove-compute-restore (RCR) method for geoid calculation. It is designed to be almost entirely automated.

- Automatically downloads [SRTM30PLUS](https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/) over the bounding box of interest
- Downloads multiple SRTM30PLUS tiles if bounding box extends over multiple tiles
- Automatically downloads a GGM from [ICGEM](https://icgem.gfz-potsdam.de/tom_longtime)
- Uses a template file so that users do not have to interact with the scripts
- Can automatically download other DEMs (e.g., Copernicus DEM, NASADEM, and GEBCO). Default is SRTM30PLUS

## Installation
`GeoidLab` can be installed using conda/mamba or pip.
  
```
conda create -n geoid_env -y
mamba install -c conda-forge geoidlab -y
```      
Test installation

```
compute_geoid.py -h
compute_reference_geoid.py -h
```
## Examples
- Prepare data: This includes all relevant data for your study area:
  - Terrestrial gravity data
  - Marine gravity data
  - Global Geopotential Model (GGM). `GeoidLab` can automatically download this. Just provide the name of the GGM in the template file
  - Digital Elevation Model (DEM). `GeoidLab` presently downloads SRTM30PLUS
- Ensure that the terrestrial gravity data has columns `lon`, `lat`, `gravity`, and `height`. If you are providing gravity anomalies,
  ensure that they are free-air anomalies
- Call `compute_geoid.py --start [step]`

See the [tutorial repo](https://github.com/cikelly/geoidlab-tutorial) for examples of using `GeoidLab`.

## References
- Yakubu, C. I., Ferreira, V. G. and Asante, C. Y., (2017): [Towards the Selection of an Optimal Global Geopotential
Model for the Computation of the Long-Wavelength Contribution: A Case Study of Ghana, Geosciences, 7(4), 113](http://www.mdpi.com/2076-3263/7/4/113)

- C. I. Kelly, S. A. Andam-Akorful, C. M. Hancock, P. B. Laari & J. Ayer (2021): [Global gravity models and the Ghanaian vertical datum: challenges of a proper definition, Survey Review, 53(376), 44–54](https://doi.org/10.1080/00396265.2019.1684006)
