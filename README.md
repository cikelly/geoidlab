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

## Installation
- easy-pygeoid can be installed using conda/mamba or pip.
  
```
conda create -n geoid_env -y
mamba install -c conda-forge easy-pygeoid -y
```      
- Test installation

```
compute_geoid.py -h
compute_reference_geoid.py -h
```
  
## References
- Yakubu, C. I., Ferreira, V. G. and Asante, C. Y., (2017): [Towards the Selection of an Optimal Global Geopotential
Model for the Computation of the Long-Wavelength Contribution: A Case Study of Ghana, Geosciences, 7(4), 113,](http://www.mdpi.com/2076-3263/7/4/113)

- Featherstone, W.E. (2003): [Software for computing five existing types of deterministically modified integration kernel for gravimetric geoid determination, Computers & Geosciences, 29(2)](http://linkinghub.elsevier.com/retrieve/pii/S0098300402000742)

- Holmes, S.A. and Featherstone, W.E., (2022): [A unified approach to the Clenshaw summation and the recursive computation of very high degree and order normalised associated Legendre functions, Journal of Geodesy, 76(5), 279-299](https://link.springer.com/article/10.1007/s00190-002-0216-2)
