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
**The datasets used for this methodology are not redistributed through this repository, though they are needed to run the scripts**. Please refer to the respective publications for access and licensing information. A preprocessing file has been included that follows the original data structure as obtained from these links:

- Ice thickness and surface elevation data (Grab et al., 2020): https://doi.org/10.3929/ethz-b-000434697
- Ice velocity data (Millan et al., 2022): https://doi.org/10.6096/1007
- Elevation change data (Hugonnet et al., 2021): https://doi.org/10.6096/13.
- GLAMOS glacier mass balance observations: https://doi.org/10.18750/MASSBALANCE.POINT.2021.R2021 and https://doi.org/10.18750/massbalance.2023.r2023
- Glacier outlines, from the Randolph Glacier Inventory (RGIv6, https://doi.org/10.7265/4M1F-GD79; RGI Consortium, 2017) and Swiss Glacier Inventory (SGI2016, doi: https://doi.org/10.3389/feart.2021.704189; Linsbauer et al., 2021).

### Output data
The results of this study, distributed (2D) Mass Balance estimates for 100 Swiss glaciers, will be published open access.
[ link and DOI to be included here ]

## Repository structure

`data/`
- `bestParams/` : directory with (final) output of methodology
  - `mb_2000-2020/`: all derived 2D Mass Balance files for the 2000-2020 period using best performing hyperparameters.
  - `monteCarlo/`: output of 100 monte carlo runs, only for two demo glaciers, can be used for plotting scripts.
  -`aggregated_glacier_values_2000-2020.csv`: file with aggregated output values of the method per glacier for the 2000-2020 period, including average glacier mass balance (MB), the median MB uncertainty across the glacier, ELA, AAR and mass balance gradients.
- `rmse2glamos`: performance values (RMSE) of different filter approaches after running all approach+hyperparameter combinations.

`figures/`: example figures for two demo glaciers (Aletsch & Findel) that can be produced with the scripts and are also in the manuscript.

`files/`
  - `best_parameters_bruteforce_weighted.json`: file with chosen optimal hyperparameters
  - `glamos_sgi_name-matches.csv`: lookup table for GLAMOS glacier filename for point MB.
  - `glamos_train_test_split.csv`: Table 1 in manuscript

`scripts/`: main scripts deriving distributed mass balance. Main processes is referred to as "Brute force approach", where 246 realizations are created per glacier across different approach & hyperparameter space. The process is split into multiple files (for clarity of processing and being able to easily re-do intermediate steps). The files require the availability of all input data (see section above for download).
The following files are run in sequence:
- `bruteForce1.py`:
    Calculates 2D flux divergence for each filtering approach (f000, f001, f110, f111) and all parameter combinations; 2D fluxdiv fields stored as .tif per realization, per glacier (temporary output)
- `bruteForce2.py`:
    Converts fluxdiv to mass balance and applies density conversion (to m.w.e./yr). 2D fluxdiv and MB assembled into one netcdf per glacier containing all realizations(intermediate output)
- `bruteForce3.py`:
    Evaluates the MB from all approaches & hyperparameter space to GLAMOS stake and elevation-binned mass balance data. Saves evaluation metrics (RMSE) to netcdf (N,F,glacier)
- `bruteForce4.py`:
    Plots the evaluation metrics (RMSE) per approach, which is used (by authors) to select the best parameter combination for each approach.
    Saves all glacier RMSE for the bestParam set in an excel (supplementary Table 1 in manuscript).
    Extract and saves the (final) output (flux div & mass balance) of each glacier for the chosen best-parameter combination (.tiff).

`scripts/plots/`: files to reproduce (part of) the figures in the manuscript. Some scripts can run with the available (demo) data in this repository, others require the full output of the 'bruteForce' approach. NB: Not all manuscript figures are plotted with python; the 2D MB maps are visualized using QGIS.

`scripts/postprocessing/`: extracting all relevant glaciological values from the produced 2D MB maps: aggregated statistics, Equilibrium line altitude (ELA), Accumulation-area-ratio (AAR), and Mass Balance gradients (seperately calculated for the ablation/accumulation area).

`scripts/preprocessing/`:
  - `preprocessing_downsample_to_50m.py`: assembling all input datasets to the same grid, CRS and resolution. The resulting files are used for all other scripts.
  - `glacier_split_train-test_set.py`: defining which GLAMOS glaciers go into 'train' and 'test' set for methdology development. Table 1 in manuscript.



## Citation

If you use this code, please cite: [to be added]

The archived version of this repository is available through Zenodo: **[ZENODO DOI to be added]**

## License

MIT License
