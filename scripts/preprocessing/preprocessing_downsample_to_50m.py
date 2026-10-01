#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:22:03 2025

@author: M Izeboud

-------------------------
read and save doonwsampled GRAB thickness & elevation
-------------------------
"""
#%%
from pathlib import Path

import numpy as np
import rasterio as rio
import xarray as xr
import myFunctions as myf
import os

data_dir = Path('/Users/mizeboud/Documents/Data_iCloud/SMB2D/')
target_crs = 'EPSG:32632' ## using EPSG:32632 (UTM32N), same as the Millan velocity data, to avoid reprojecting velocity vectors.
target_resolution_m = 50


def reproject_and_save( datafile, resolution_m, target_crs='EPSG:32632',
                        variable_name=None, match_ref_grid=None,
                        resampling=rio.enums.Resampling.bilinear,
                        save_tiff = False):
    """Reproject a raster and save it beside its input with a descriptive name.

    Output naming is ``[variable]_[resolution-m]_epsg[crs].tif``. 
    If ``match_grid`` is supplied, match its exact extent and pixel alignment.
    """
    datafile = Path(datafile)
    variable_name = variable_name or datafile.stem
    print('Varname: ', variable_name)

    with xr.open_dataarray(datafile) as source:
        da_data = source.isel(band=0) if 'band' in source.dims else source
        da_data = da_data.load()
    assert da_data.rio.crs is not None, f'Input {datafile} has no CRS.'
    
    if match_ref_grid is None: ## no reprojecting, clip to swiss domain. Only for data that is already in target_crs (only velocity)
        bbox_swiss_32632 = [ 243071.95534342, 5049658.05308324,  634818.83410692, 5331921.25252423] # in EPSG:32632; including small buffer
        assert da_data.rio.crs.to_string() == target_crs, f'Input {datafile} is not in target CRS {target_crs}.'

        # clip global-velocity by Swiss bounds
        xmin, ymin, xmax, ymax = bbox_swiss_32632 
        da_out = da_data.rio.clip_box( # clip using bounds in a CRS different from the dataset
            minx=xmin,  miny=ymin,
            maxx=xmax,  maxy=ymax,
            crs="EPSG:32632", # bounds are in this CRS, but will clip dataset in the dataArray's own CRS (32632 for VX)
        )
    else: ## reproject to match the reference grid
         da_out = myf.reproject_match_grid( match_ref_grid , da_data , resample_method=resampling)

    da_out = da_out.rename(variable_name)
    crs_print = target_crs.split(':')[-1] if ':' in target_crs else target_crs
    output_file = os.path.join(os.path.dirname(datafile), f'{variable_name}_{resolution_m}m_epsg{crs_print}.tif')
    # output_file = datafile.parent / f'{variable_name}_{resolution_m}m_epsg{target_crs}.tif'  
    print(output_file)
    if save_tiff:
        da_out.rio.to_raster(output_file)
    return da_out, output_file


''' -------------------------
## Millan et al. 2022: velocities
# 50 m, 1 yr (2017-18); 98% global glaciers
# EPSG 32632 
--> this is the base grid. No need to reproject, but do clip to Swiss domain.
VX, VY, STDX, STDY
------------------------- '''
path2data = os.path.join(data_dir,'GlobalGlacierVelocity_Millan2022/RGI-11/')
vx_file = 'VX_RGI-11_2021July01.tif'
vy_file = 'VY_RGI-11_2021July01.tif'
vx_err_file = 'STDX_RGI-11_2021July01.tif'
vy_err_file = 'STDY_RGI-11_2021July01.tif'

## example for one of the files
da_vx_swiss, vx_fileOut = reproject_and_save(os.path.join(path2data, vx_file), target_resolution_m, target_crs='EPSG:32632',
                        variable_name=None, match_ref_grid=None, #match_ref_grid=None means no reprojecting but only clipping to swiss domain.
                        resampling=rio.enums.Resampling.bilinear,
                        save_tiff = False ## toggle to save file
                        )

#%%
''' -------------------------
## Grab et al. 2021: ice thickness
# 10 m resolution, annual; covers 81% of Swiss glaciers
# EPSG 2056
--> downsampled to 50 m of Millan and reproject to EPSG32632

Thickness, thickness errors (minus and plus), Elevation
------------------------- '''


# Grab et al. 2021 ice thickness: 10 m, EPSG:2056.

file_thickness = os.path.join(data_dir,'SwissGlacierThickness-R2020/04_IceThickness_SwissAlps/IceThickness.tif')
file_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/SwissALTI3D_r2019.tif')
file_thicknessMinus = os.path.join(data_dir, 'SwissGlacierThickness-R2020/05_IceThicknessUncertainty_SwissAlps/IceThicknessUncertaintyMinus.tif')
file_thicknessPlus = os.path.join(data_dir, 'SwissGlacierThickness-R2020/05_IceThicknessUncertainty_SwissAlps/IceThicknessUncertaintyPlus.tif')

## example for thickness; replace file for elevation, thicknesMinus and thicknessPlus
da_thickness_50m, thickness_50m_file = reproject_and_save(file_thickness, target_resolution_m, target_crs='EPSG:32632',
                        variable_name=None, ## if None: uses filename_in + info for saving filename
                        match_ref_grid=da_vx_swiss,
                        resampling=rio.enums.Resampling.bilinear,
                        save_tiff = False ## toggle to save file
)

#%%

''' -------------------------
## Load Hugonnet et al. 2021 dHdt : 
# 100 m resolution, saved for tiles with varying CRS (EPSG:32632, EPSG:32633)

--> reproject and combine to regional file
------------------------- '''

def load_combine_files_to_regional(filelist_tiles, da_to_match, resample=rio.enums.Resampling.bilinear):
    """Load and combine multiple raster files to a regional grid.
    
    If data tiles have gaps, that's an issue with the combining/reprojecting or loading all files directly with xr.open_mfdataset, giving strange artifacts
    Instead, each datatile needs to be loaded and reprojected first to the whole domain (just filling with empty nans) 
        so then the stack of all the same domain can be combined.
    """
    da_list = []
    for tile in filelist_tiles:
        # load individual file and reproject to swiss domain
        with xr.open_dataarray(tile).isel(band=0) as da_tmp:
            da_reprj = myf.reproject_match_grid(da_to_match, da_tmp, resample_method=resample)
            da_list.append(da_reprj)
    
    # Combine all reprojected data arrays into one
    combined_da = da_list[0] ## select first to initialize the combined data array
    for ds in da_list[1:]:
        combined_da = combined_da.combine_first(ds)
    
    return combined_da
save_tiff = False ## toggle for saving
tiles_swiss = ['N45E006','N45E007','N46E006','N46E007','N46E008','N46E009','N46E010','N47E008','N47E009','N47E010']

''' ## dhdt for different periods '''
path2dhdt = os.path.join(data_dir,'GlacierElevationChange_Hugonnet2021/11_rgi60_2000-01-01_2020-01-01/dhdt/')
path2dhdt = os.path.join(data_dir,'GlacierElevationChange_Hugonnet2021/11_rgi60_2010-01-01_2020-01-01/dhdt/')
path2dhdt = os.path.join(data_dir,'GlacierElevationChange_Hugonnet2021/11_rgi60_2015-01-01_2020-01-01/dhdt/')

## select one of the dhdt periods to process
dhdt_hugo_swiss = load_combine_files_to_regional(filelist_tiles=[os.path.join(path2dhdt, tile + '.tif') for tile in tiles_swiss],
                                                 da_to_match=da_vx_swiss)


if save_tiff:
    save_filename = os.path.join(path2dhdt, 'dHdt_swiss_50m_epsg32632.tif')
    dhdt_hugo_swiss.rio.to_raster(save_filename)


''' ## dhdt err for different periods '''
path2dhdt_err = os.path.join(data_dir,'GlacierElevationChange_Hugonnet2021/11_rgi60_2000-01-01_2020-01-01/dhdt_err/')
path2dhdt_err = os.path.join(data_dir,'GlacierElevationChange_Hugonnet2021/11_rgi60_2010-01-01_2020-01-01/dhdt_err/')
path2dhdt_err = os.path.join(data_dir,'GlacierElevationChange_Hugonnet2021/11_rgi60_2015-01-01_2020-01-01/dhdt_err/')

## select one of the dhdt periods to process
dhdt_hugo_swiss = load_combine_files_to_regional(filelist_tiles=[os.path.join(path2dhdt_err, tile + '.tif') for tile in tiles_swiss],
                                                 da_to_match=da_vx_swiss)

if save_tiff:
    save_filename = os.path.join(path2dhdt_err, 'dHdt_err_swiss_50m_epsg32632.tif')
    dhdt_hugo_swiss.rio.to_raster(save_filename)



#%%
''' -------------------------
## Farinotti et al. 2019 ice thickness
# XX resolution, saved for individual glacier files

--> combine to regional file
------------------------- '''

files_f2019 = os.listdir(os.path.join(data_dir,'thickness_farinotti2019/composite_thickness_RGI60-all_regions/RGI60-11/'))
files_f2019.sort()
files_f2019
filelist_f2019 = [os.path.join(data_dir,'thickness_farinotti2019/composite_thickness_RGI60-all_regions/RGI60-11/', filename) for filename in files_f2019]

## open indiviudal glaciers and combine to swiss-wide grid

da_farinot = load_combine_files_to_regional(filelist_tiles=filelist_f2019,
                                            da_to_match=da_vx_swiss,
                                            resample=rio.enums.Resampling.bilinear)

if save_tiff:
    # save tiff
    da_farinot.rio.to_raster(os.path.join(data_dir,'thickness_farinotti2019/IceThickness_Swiss_50m_epsg32632.tif'))
