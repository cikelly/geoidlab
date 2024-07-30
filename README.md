# easy-pygeoid

easy-pygeoid is a Python package developed for easy geoid modelling. 
The package comes packed with utilities for estimating a geoid model using Stokes' method, with options for:

- the original Stokes' kernel
- Wong and Gore's modification of Stokes' kernel
- Heck and Gruninger's modification of Stokes' kernel
- Terrain correction

easy-pygeoid uses the remove-compute-restore (RCR) method for geoid calculation. It is designed to be almost entirely automated.

- Automatically downloads [SRTM30PLUS](https://topex.ucsd.edu/pub/srtm30_plus/srtm30/grd/) over the bounding box of interest
- Automatically downloads a GGM from [ICGEM](https://icgem.gfz-potsdam.de/tom_longtime)
- Uses a template file so that users do not have to interact with the scripts
