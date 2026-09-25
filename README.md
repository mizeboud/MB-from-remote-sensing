# MB-from-remote-sensing
This repository contains the code used to derive spatially distributed glacier mass balance (MB) estimates for the Swiss Alps using the mass-continuity equation.
The code accompanies the manuscript "Deriving Distributed Mass Balance Patterns of Swiss Glaciers from Remote Sensing Data" by Izeboud et al. (submitted to The Cryosphere)

Maaike Izeboud\
Vrije Universiteit Brussel

## Overview and Method

The approach derives distributed glacier mass balance (MB) from (distributed) estimates of:

* glacier surface velocity (vx, vy)
* ice thickness
* surface elevation change

Ice flux divergence is calculated from the ice thickness and surface velocity fields and combined with observed elevation change to retrieve spatially distributed MB. Spatial filtering is applied to reduce local noise introduced by uncertainties in the input datasets and their spatial gradients. Multiple approaches were tested, which were evaluated against GLAMOS observations to choose the best setup.

The chosen best approach and parameters are used to create a regional distributed MB product of 100 glaciers in Swiss Alps. For a complete description of the methodology, filtering procedure, parameter selection, evaluation, and uncertainty assessment, please refer to the accompanying manuscript.


## Input datasets
The datasets used for this methodology are not redistributed through this repository, though they are needed to run the scripts. Please refer to the respective publications for access and licensing information.

- Ice thickness and surface elevation data (Grab et al., 2020): https://doi.org/10.3929/ethz-b-000434697
- Ice velocity data (Millan et al., 2022): https://doi.org/10.6096/1007
- Elevation change data (Hugonnet et al., 2021): https://doi.org/10.6096/13.
- GLAMOS glacier mass balance observations: https://doi.org/10.18750/MASSBALANCE.POINT.2021.R2021 and https://doi.org/10.18750/massbalance.2023.r2023
- Glacier outlines, from the Randolph Glacier Inventory (RGIv6, https://doi.org/10.7265/4M1F-GD79; RGI Consortium, 2017) and Swiss Glacier Inventory (SGI2016, doi: https://doi.org/10.3389/feart.2021.704189; Linsbauer et al., 2021).


## Repository structure

`files/` – \
`scripts/` – main scripts deriving distributed mass balance\
`scripts/preprocessing` – preprocessing of the input datasets\
`scripts/postprocessing` –


## Citation

If you use this code, please cite: [to be added]

The archived version of this repository is available through Zenodo: **[ZENODO DOI to be added]**

## License

MIT License
