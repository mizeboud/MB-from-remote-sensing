## 2D SMB estimation for swiss glacier
# Brute force approach: calculate for multiple parameter set.
# Process and save output; evaluation at later stage.

# author: M Izeboud
# July 2025


# %% Imports

import xarray as xr
import rasterio as rio
from rioxarray.exceptions import NoDataInBounds
import numpy as np
import os 
import geopandas as gpd
from shapely.geometry import Polygon

import itertools
from joblib import Parallel, delayed

import time
from tqdm import tqdm
import gc

import pandas as pd

import myFunctions as myf

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), used for all processing (to avoid reprojecting velocity vectors)
data_dir = '../data/' ## data directory, where input data is stored and output will be saved

## TMP:
homedir = '/Users/mizeboud/Documents/Documents_mizeboud/PostDoc/2D-SMB/'
data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'

#%% Functions


def prepare_glacier(glaciers, glacier_index, da_grab, ds_vx, ds_vy,
                    target_crs, output_base):
    """Extract one glacier's input fields and calculate its gradients.
    - clip thickness and velocity to glacier outline
    - calculate gradients of thickness and velocity (handle NaNs at glacier edges)
    - create output directory for glacier

    Returns a dictionary with glacier data and gradients, or None if the glacier has insufficient data.
    {
        'glacier_rgiid' : RGI ID of the glacier,
        'output_dir'    : path to the output directory,
        'da_thickness'  : thickness data,
        'da_vx'         : x-component of velocity,
        'da_vy'         : y-component of velocity,
        'gradients'  : calculated gradients in a list, [dHdx, dHdy, dudx, dudy, dvdx, dvdy]
    }
    """
    glacier = glaciers.iloc[[glacier_index]]
    glacier_rgiid = glacier['RGIId'].item()
    output_dir = os.path.join(output_base, glacier_rgiid)

    try:
        da_thickness = da_grab.rio.clip(
            glacier.geometry, drop=True, all_touched=True)
        da_vx = ds_vx.rio.clip(
            glacier.geometry, drop=True, all_touched=True)['vx']
        da_vy = ds_vy.rio.clip(
            glacier.geometry, drop=True, all_touched=True)['vy']
    except NoDataInBounds as error:
        print(f"No raster data in bounds for glacier {glacier_rgiid}: {error}")
        return None

    if xr.where(~np.isnan(da_thickness), 1, 0).sum() <= 1:
        return None

    for approach in ('f000', 'f001', 'f110', 'f111'):
        os.makedirs(os.path.join(output_dir, approach), exist_ok=True)

    gradients = myf.wrap_calc_centraldiff_gradients(
        da_thickness, da_vx, da_vy,
        glacier, target_crs,
        extrapolate_method='nanmask',
    )

    return {
        'glacier_rgiid': glacier_rgiid,
        'output_dir': output_dir,
        'da_thickness': da_thickness,
        'da_vx': da_vx,
        'da_vy': da_vy,
        'gradients': gradients,
    }


def process_param_grid(
        lscale_gradients, lscale_fdiv, fparam, 
        dHdx, dHdy, dudx, dudy, dvdx, dvdy,
        da_thickness, da_vx, da_vy, 
        max_ksize_pxs,
        glacier_rgiid,
        nthreads=2,
        ):
    '''Parallel calculation of flux divergence for each approach & hyperparameter value.
    Returns a dictionary with the results for the given parameter combination.
    '''

    import numba
    numba.set_num_threads(nthreads)

    fparam = round(fparam,2) # to adjust floating point precision issues that were introduced by itertools.paramgrid

    # 1. Gradient smoothing
    dHdx_smooth = myf.exp_smooth_numba(dHdx, da_thickness, n_thickn_scale=lscale_gradients, min_ksize=2, max_ksize=max_ksize_pxs)
    dHdy_smooth = myf.exp_smooth_numba(dHdy, da_thickness, n_thickn_scale=lscale_gradients, min_ksize=2, max_ksize=max_ksize_pxs)
    dudx_smooth = myf.exp_smooth_numba(dudx, da_thickness, n_thickn_scale=lscale_gradients, min_ksize=2, max_ksize=max_ksize_pxs)
    dudy_smooth = myf.exp_smooth_numba(dudy, da_thickness, n_thickn_scale=lscale_gradients, min_ksize=2, max_ksize=max_ksize_pxs)
    dvdx_smooth = myf.exp_smooth_numba(dvdx, da_thickness, n_thickn_scale=lscale_gradients, min_ksize=2, max_ksize=max_ksize_pxs)
    dvdy_smooth = myf.exp_smooth_numba(dvdy, da_thickness, n_thickn_scale=lscale_gradients, min_ksize=2, max_ksize=max_ksize_pxs)

    # 2. calculate  fluxdiv for each filter approach and scale fluxdiv for mass conservation to f000 

    # 2. f000
    fluxdiv000 = myf.calc_flux_div(da_thickness, dHdx, dHdy, 
                                    da_vx, dudx, dudy, 
                                    da_vy, dvdx, dvdy, fparameter=fparam)


    # 2. f001
    fluxdiv001_raw = myf.exp_smooth_numba(fluxdiv000, da_thickness, n_thickn_scale=lscale_fdiv, min_ksize=2, max_ksize=max_ksize_pxs)
    fluxdiv001, fdiv_ratio001 = myf.scale_fluxdiv_uniform(fluxdiv001_raw, fluxdiv000)
    

    # 2. f110
    fluxdiv110_raw = myf.calc_flux_div(da_thickness, dHdx_smooth, dHdy_smooth, 
                                    da_vx, dudx_smooth, dudy_smooth, 
                                    da_vy, dvdx_smooth, dvdy_smooth, fparameter=fparam)
    fluxdiv110, fdiv_ratio110 = myf.scale_fluxdiv_uniform(fluxdiv110_raw, fluxdiv000)

    if lscale_gradients < 6 and lscale_fdiv < 6: 
        # only calculate f111 if both smoothing scales are below 6 
        # (dont want to calculate for higher scales, so can save time by skipping)
        fluxdiv111_raw = myf.exp_smooth_numba(
            fluxdiv110_raw,
            da_thickness,
            n_thickn_scale=lscale_fdiv,
            min_ksize=2,
            max_ksize=max_ksize_pxs,
        )
        fluxdiv111, fdiv_ratio111 = myf.scale_fluxdiv_uniform(fluxdiv111_raw, fluxdiv000)
    else:
        fluxdiv111 = None

    return {
        'lscale_gradients': lscale_gradients,
        'lscale_fdiv': lscale_fdiv,
        'fparam': fparam,
        'fluxdiv000': fluxdiv000,
        'fluxdiv001': fluxdiv001,
        'fluxdiv110': fluxdiv110,
        'fluxdiv111': fluxdiv111,
        # 'fdiv_ratio001': fdiv_ratio001,
        # 'fdiv_ratio110': fdiv_ratio110,
        # 'fdiv_ratio111': fdiv_ratio111,
    }


def save_fluxdiv_results(results, glacier_rgiid, output_dir):
    """Save one calculated parameter combination in the standard layout."""
    fparam_str = f"{results['fparam']:.2f}".replace('.', '')
    lscale_gradients = results['lscale_gradients']
    lscale_fdiv = results['lscale_fdiv']

    output_paths = {
        'fluxdiv000': os.path.join(output_dir, 'f000', f'{glacier_rgiid}_fluxdiv_f000_F{fparam_str}_N0.tif'),
        'fluxdiv001': os.path.join(output_dir, 'f001', f'{glacier_rgiid}_fluxdiv_f001_F{fparam_str}_N{lscale_fdiv}.tif'),
        'fluxdiv110': os.path.join(output_dir, 'f110', f'{glacier_rgiid}_fluxdiv_f110_F{fparam_str}_N{lscale_gradients}.tif'),
        'fluxdiv111': os.path.join(output_dir, 'f111', f'{glacier_rgiid}_fluxdiv_f111_F{fparam_str}_Ng{lscale_gradients}-Nf{lscale_fdiv}.tif'),
    }
    for result_name, output_path in output_paths.items():
        if results[result_name] is None:
            continue
        results[result_name].rename('fluxdiv').rio.to_raster(output_path)
#%%

''' ################################################################

(1) Load regional (swiss) data including all glaciers to process

#################################################################### '''


'''------------------
## Glacier outlines (RGI shapefiles)
------------------'''
swiss_bounds = [ 243071.95534342, 5049658.05308324,  634818.83410692, 5331921.25252423] ## In EPSG:32632; including small buffer
swiss_bounds_poly = gpd.GeoDataFrame(index=[0], crs=target_crs, geometry=[Polygon.from_bounds(*swiss_bounds)])  

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs, box_bounds=swiss_bounds)


'''------------------
## Load thickness Grab et al. 2021
# 10 m resolution, annual; covers 81% of Swiss glaciers
# EPSG:32632 
# --> downsampled to 50 m of Millan and reproject to EPSG32632
------------------'''

## load preprocessed reprojected file
file_grab_thickness = os.path.join(data_dir,'SwissGlacierThickness-R2020/04_IceThickness_SwissAlps/','IceThickness_50m_epsg32632.tif') # already in EPSG:32632 
da_grab = xr.open_dataarray(file_grab_thickness).isel(band=0)


''' -------------------
## Millan et al. 2022: velocities
# 50 m, 1 yr (2017-18); 98% global glaciers
# EPSG 32632 --> use this projection and resolution, as it has potential for global application
-------------------'''
path2data = os.path.join(data_dir,'GlobalGlacierVelocity_Millan2022/RGI-11/')
vx_file = 'VX_RGI-11_2021July01.tif'
vy_file = 'VY_RGI-11_2021July01.tif'

# clip global-velocity by Swiss bounds
xmin, ymin, xmax, ymax = swiss_bounds 
with xr.open_dataset(os.path.join(path2data,vx_file)) as ds_tmp:
    ds_tmp = ds_tmp.rename({'band_data':'vx'}).isel(band=0)
    ds_vx = ds_tmp.rio.clip_box( # clip using bounds in a CRS different from the dataset
        minx=xmin,  miny=ymin,
        maxx=xmax,  maxy=ymax,
        crs=swiss_bounds_poly.crs, ## specify in which CRS the bounds are given
    )
with xr.open_dataset(os.path.join(path2data,vy_file)) as ds_tmp:
    ds_tmp = ds_tmp.rename({'band_data':'vy'}).isel(band=0)
    ds_vy = ds_tmp.rio.clip_box( # clip using bounds in a CRS different from the dataset
        minx=xmin,  miny=ymin,
        maxx=xmax,  maxy=ymax,
        crs=swiss_bounds_poly.crs,
    )
del ds_tmp


'''-----------------
## Set up parameter grid  and choose workflow
## choose STEP 1 or STEP 2, but both need to be run to have complete set of simulations
---------------------'''

param_grid, max_ksize_pxs = myf.make_parameter_grid()
print(f"Total parameter combinations: {len(param_grid)}")


#%%
''' --------------
LOOP GLACIERS
------------------'''

t0 = time.time()
# for gi in tqdm(range(len(gdf_swiss_large)), desc="Processing glaciers"):  ## process all
# for gi in tqdm(range(0, 3), desc="Processing glaciers"):  ## process only first glacier for testing
for gi in tqdm(range(len(gdf_swiss_large)-1, len(gdf_swiss_large)), desc="Processing glaciers"):  ## process last glacier (claridenfirn)

    '''##########################################
    Prep glacier data: 
    - clip to RGI outline, 
    - calculate gradients (central diff scheme), 
    - intialize output dir for every glacier
    #############################################'''
    glacier_data = prepare_glacier(
        gdf_swiss_large, gi,   # geoDataFrame and index of glacier to process
        da_grab, ds_vx, ds_vy, # thickness, velocity-x , velocity-y
        target_crs,            # CRS to use for processing
        output_base=os.path.join(data_dir, 'bruteForceTMP', 'glaciers'),
    )
    if glacier_data is None:
        continue

    glacier_rgiid = glacier_data['glacier_rgiid']
    path2save_glacier = glacier_data['output_dir']
    da_thickness = glacier_data['da_thickness']
    da_vx = glacier_data['da_vx']
    da_vy = glacier_data['da_vy']
    dHdx, dHdy, dudx, dudy, dvdx, dvdy = glacier_data['gradients']

    '''##########################################
    brute force process param_grid  (with progress bar)
    #############################################'''

    njobs_glacier = 3 # number of parallel processing of parameter-grid for current glacier
    nthread_numba = 3 # number for parallel smoothing function
    total_cpu = nthread_numba * njobs_glacier
    if total_cpu > 10: # 12 CPUs available
        raise Warning('Total number of threads ({total_cpu}) exceeds 10, make sure enough CPUs are availalbe.')
    
    results = Parallel(n_jobs=njobs_glacier)( 
        delayed(process_param_grid)( # returns dictionary with fdiv_f000 etc results for each parameter combination
            lscale_gradients, lscale_fdiv, fparam,
            dHdx, dHdy, dudx, dudy, dvdx, dvdy,
            da_thickness, da_vx, da_vy,
            max_ksize_pxs,
            glacier_rgiid,
            nthreads=nthread_numba
        )
        for lscale_gradients, lscale_fdiv, fparam in param_grid
    )

    ''' ############################ 
    save results for current glacier 
    ################################'''
    for result in results:
        save_fluxdiv_results(result, glacier_rgiid, path2save_glacier)

    # Delete large arrays to free memory
    del dHdx, dHdy, dudx, dudy, dvdx, dvdy
    del da_thickness, da_vx, da_vy, 
    gc.collect()

print('---- \n DONE ; total {:.3f}min'.format((time.time()-t0)/60))

# %%
