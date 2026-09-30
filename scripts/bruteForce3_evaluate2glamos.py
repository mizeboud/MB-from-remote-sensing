'''
Brute force approach: calculate 2D distributed MB estimates for multiple approaches and parameter set.
The process is split into multiple files (for clarity of processing and being able to re-do intermediate steps)
The following files are run in sequence:

bruteForce1: 
    Calculates 2D flux divergence for each filtering approach (f000, f001, f110, f111) and all parameter combinations; 
    2D fluxdiv fields stored as .tif per realization, per glacier (temporary output)
bruteForce2: 
    Converts fluxdiv to mass balance and applies density conversion (to m.w.e./yr)
    2D fluxdiv and MB assembled into one netcdf per glacier containing all realizations(intermediate output)
bruteForce3: 
    Evaluates the MB from all approaches & hyperparameter space to GLAMOS stake and elevation-binned mass balance data. 
    Saves evaluation metrics (RMSE) to netcdf (N,F,glacier)
bruteForce4: 
    Plots the evaluation metrics (RMSE) per approach, which is used (by authors) to select the best parameter combination for each approach.
    Saves all glacier RMSE for the bestParam set in an excel (supplementary Table 1 in manuscript).
    Extract and saves the (final) output (flux div & mass balance) of each glacier for the chosen best-parameter combination (.tiff).

# author: M Izeboud
# January 2026'''

#%%
import ast
import warnings

import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import geopandas as gpd

import myFunctions as myf
import evalFunctions as evalF

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes
data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
# data_dir = '../data/' ## data directory, where input data is stored and output will be saved
path2glacier_output = os.path.join(data_dir,'bruteForceTMP/glaciers/')
# path2glacier_output = '../data/bruteForce/' ## dir with demo data
path2saveRSME = os.path.join(data_dir, 'bruteForceTMP/rmse2glamos/')


#%%


''' ################################################################
Load SGI and RGI info
#################################################################### '''


'''## RGI inventory to mask glacier areas'''

gl_outline_swiss = myf.load_rgi_outlines_swiss(
    filepath=os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
    area_km2=2, target_crs=target_crs)




''' ################################################################
Grab thickness & elevation data
#################################################################### '''

## load preprocessed reprojected file

file_grab_thickness = os.path.join(data_dir,'SwissGlacierThickness-R2020/04_IceThickness_SwissAlps/IceThickness_50m_epsg32632.tif')
da_grab = xr.open_dataarray(file_grab_thickness).isel(band=0)

file_grab_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/IceElevation_50m_epsg32632.tif') 
da_grab_elev = xr.open_dataarray(file_grab_elevation).isel(band=0)
da_grab_elev.rio.crs # EPSG 32632

#%% ## Load GLAMOS

''' ################################################################
Load GLAMOS data
#################################################################### '''

path2data_glamos = os.path.join(data_dir,'GLAMOS/massbalance_fixdate_elevationbins.csv')
df_glamos_mb = myf.load_glamos_elevationbins(path2data_glamos)

df_glamos_sgi_file = os.path.join('../files/glamos_train_test_split.csv')
df_glamos_sgi = pd.read_csv(df_glamos_sgi_file, index_col=0)
df_glamos_sgi['RGI_match'] = df_glamos_sgi['RGI_match'].apply(ast.literal_eval)

glamos_traintest = df_glamos_sgi.copy()  # SGIID as index
df_glamos_sgi.reset_index(inplace=True) ## SGIID as column


''' -----------------------
Load point-data from GLAMOS stakes for evaluation
- in CH1903 coordinate system; == EPSG 2056
----------------------- '''

## open table with matched filenames
glamos_file_names = pd.read_csv('../files/glamos_sgi_name-matches.csv')

#%%
''' ################################################################
Some processing settings
#################################################################### '''

save_nc = False # can toggle for testing/debugging

N_values = np.arange(0,11,1)
F_values = np.arange(0.75, 1.01, 0.05) # e.g., 0.75, 0.80, ..., 1.0

''' ################################################################
Loop GLAMOS glaciers
#################################################################### '''

for glacier_sgiid in glamos_traintest.index[[0,9]]: ## Aletsch & Findel; demo glaciers
    ## glacier gdf
    df_glacier_sgi = df_glamos_sgi.loc[df_glamos_sgi['sgi-id']== glacier_sgiid] # select a row with multiple RGI  matches
    glacier_name = df_glacier_sgi['name'].item()

    print('-----------\n', glacier_sgiid, glacier_name )


    '''## select a single glacier to evaluate'''
    glacier_stakefile = glamos_file_names.loc[glamos_file_names['SGIID']==glacier_sgiid]['stake_csv'].item()

    '''--------------------
    ## Find the RGI-id match
    --------------------'''

    rgi_matches = df_glacier_sgi['RGI_match'].item()
    if glacier_sgiid == 'A50i-19':
        ## manual fix for Claridenfirn, as i've merged the 4 RGIs it consists of
        glacier_rgiid = 'RGI60-11.008merged'
    else:
        glacier_rgiid = rgi_matches[0]
    gdf_glacier_rgi = gl_outline_swiss.loc[gl_outline_swiss['RGIId'] == glacier_rgiid].copy()

    if not os.path.isdir(os.path.join(path2glacier_output, glacier_rgiid)):
        print('No output-dir available yet for ', glacier_rgiid)
        continue

    '''## Extract glacier elevation data for the RGI match (to the fluxdiv/smb calculation) and the RGI outlines'''

    ## Extract glacier elevation data and the RGI outlines
    da_elev = da_grab_elev.rio.clip( gdf_glacier_rgi.geometry, drop=True,all_touched=True) ## still in EPSG 32632
    da_thickness = da_grab.rio.clip( gdf_glacier_rgi.geometry, drop=True,all_touched=True)

    gdf_glacier_rgi.to_crs(swiss_crs ,inplace=True) ## to EPSG:21781
    
    ''' ################################################################
    Load output data from brute-force runs
    --> will need to make a 'best choice' selection of parameter space
    --> this stakes computes RMSEs for all param_grid combinations, for both w.r.t GLAMOS Elev.Bins and Stakes (point MB).
    #################################################################### '''

    path2glacier = os.path.join(path2glacier_output, glacier_rgiid)

    ''' --------------------
    ######  select which timeperiod for calculations
    --------------------'''

    glamos_y0 = 2000; dhdt_period = '2000-2020'
    # glamos_y0 = 2010; dhdt_period = '2010-2020'
    # glamos_y0 = 2015; dhdt_period = '2015-2020'

    print(f'.. using dhdt period {dhdt_period}')

    ## open f000-f001-f110 file
    ds_glacier_mb = xr.open_dataset(
        os.path.join(path2glacier, 'mb', f'{glacier_rgiid}_mb_{dhdt_period}_paramgrid.nc' ))
    ds_glacier_mb.rio.write_crs('EPSG:32632', inplace=True)

    ## open f111 file
    ds_glacier_mb_f111 = xr.open_dataset(
        os.path.join(path2glacier, 'mb', f'{glacier_rgiid}_mb-f111_{dhdt_period}_paramgrid.nc' ))
    ds_glacier_mb_f111.rio.write_crs('EPSG:32632', inplace=True)

    '''#############################################################
    1. Calculate RMSE wrt elevation-bins
    ################################################################'''

    ''' --------------------
    ## Bin my own calculated MB using Grab. elevation bins
    ds_glacier_mb is (N,F,y,x)
    mb_bin_means_f000 is (N, F, h_bin)
    -------------------- '''
    
    mb_bin_means_f000, hmin_binned, _, _ = myf.bin_da_to_elevbins(ds_glacier_mb['mb_f000'], da_elev)
    mb_bin_means_f001, hmin_binned, _, _ = myf.bin_da_to_elevbins(ds_glacier_mb['mb_f001'], da_elev)
    mb_bin_means_f110, hmin_binned, _, _ = myf.bin_da_to_elevbins(ds_glacier_mb['mb_f110'], da_elev)
    mb_bin_means_f111, hmin_binned, _, _ = myf.bin_da_to_elevbins(ds_glacier_mb_f111['mb_f111'], da_elev)

    ''' --------------------
    # Extract GLAMOS elevation bins data for this glacier
    --------------------'''

    '''## GLAMOS mass balance data'''
    ## select GLAMOS data to match dhdt period
    df_glamos_yyyy = df_glamos_mb.loc[ (df_glamos_mb['date_start'].dt.year >= glamos_y0) & \
                                       (df_glamos_mb['date_start'].dt.year <= 2020) ].copy() # only data that is in same dhdt period
    df_mb_glacier, _ = myf.get_glacier_mb_bins_glamos(df_glamos_yyyy, 
                                            glacier_id=glacier_sgiid)
    y0_bins = df_mb_glacier['date_start','min'].dt.year.min(); y1_bins = df_mb_glacier['date_start','max'].dt.year.max()
    if y0_bins < glamos_y0:
        raise ValueError(f"GLAMOS mass balance data for glacier {glacier_sgiid} starts at {df_mb_glacier['date_start','min'].dt.year.min()}, before {glamos_y0}. Check if this is intended.")

    # GLAMOS elevation bins h_bin 
    df_mb_Ba = (df_mb_glacier['Ba']/1000).copy().rename(columns={'min':'Ba_min','max':'Ba_max','mean':'Ba_mean','median':'Ba_median'})
    df_mb_Bs = (df_mb_glacier['Bs']/1000).copy().rename(columns={'min':'Bs_min','max':'Bs_max','mean':'Bs_mean','median':'Bs_median'})
    df_mb_Bw = (df_mb_glacier['Bw']/1000).copy().rename(columns={'min':'Bw_min','max':'Bw_max','mean':'Bw_mean','median':'Bw_median'})
    df_mb_eval_bins = pd.merge(df_mb_Ba, df_mb_Bs, on='h_min', how='left')
    df_mb_eval_bins = pd.merge(df_mb_eval_bins, df_mb_Bw, on='h_min', how='left')

    
    '''--------------------
    Calculate RMSE wrt elevation-bins
    - one RMSE value per N,F combination
    - no weighing applied for RMSE calculation. The binned averages are already area-weighted by the number of pixels in each bin.
    -------------------- '''
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # check overlap between mb_bin_means and glamos bins
        idx_to_select_data = np.isin(hmin_binned, df_mb_eval_bins.index.values)
        idx_to_select_glamos = np.isin(df_mb_eval_bins.index.values,hmin_binned)
        rmse_bins_f000 = np.sqrt(np.nanmean((mb_bin_means_f000[:,:,idx_to_select_data] - df_mb_eval_bins['Ba_mean'].values[idx_to_select_glamos]) ** 2, axis=-1)) # (N,F, bins) - (bins,) --> output shape (N,f)
        rmse_bins_f001 = np.sqrt(np.nanmean((mb_bin_means_f001[:,:,idx_to_select_data] - df_mb_eval_bins['Ba_mean'].values[idx_to_select_glamos]) ** 2, axis=-1))
        rmse_bins_f110 = np.sqrt(np.nanmean((mb_bin_means_f110[:,:,idx_to_select_data] - df_mb_eval_bins['Ba_mean'].values[idx_to_select_glamos]) ** 2, axis=-1))
        rmse_bins_f111 = np.sqrt(np.nanmean((mb_bin_means_f111[:,:,:,idx_to_select_data] - df_mb_eval_bins['Ba_mean'].values[idx_to_select_glamos]) ** 2, axis=-1)) # (Ng,Nf,F,bins) - (bins,) --> output shape (Ng,Nf,F)
    
    ds_rmse_elevbins = xr.Dataset({
            'rmse_f000': (('Nlength', 'Fparam'), rmse_bins_f000),
            'rmse_f001': (('Nlength', 'Fparam'), rmse_bins_f001),
            'rmse_f110': (('Nlength', 'Fparam'), rmse_bins_f110),
            'rmse_f111': (('Ngrad', 'Nfdiv', 'Fparam'), rmse_bins_f111) 
        },
        coords={
            'Nlength': N_values,
            'Fparam': F_values
        })  



    '''#############################################################
    2. Calculate RMSE wrt stakes
    - one RMSE value per N,F combination
    - For the same stake, calculate temporal mean. 
      Calculate RMSE wrt unique stakes only, but weight for the number of readings per stake.
      e.g. if one stake has 1 reading, weight=1
           and if another stake has 20 readings, weight=20
           weights will be normalized in weighted_rmse calculation.
    ################################################################'''

    ''' --------------------
    # #### get stake data of glacier
    # mb_we is in [mm w.e. / kg m-2]	; my values are in m w.e. , so divide by 1000
    # mb_error :Uncertainty of point mass balance as square root of the sum of squares of the fractional uncertainties of Density and Raw Balance [mm w.e.]) --> also divide by 1000
    --------------------'''
    
    pd_stake_data_yyyy, y0_stakes, y1_stakes = evalF.load_stake_data(
        os.path.join(
            data_dir,
            'GLAMOS/massbalance_point_2021_r2021/annual_preprocessed/',
            glacier_stakefile,
        ),
        start_year=glamos_y0,
    )

    # assert y0_bins == y0_stakes and y1_bins == y1_stakes, \
    #     f"Mismatch in time periods between elevation-bins and stake data: bins=({y0_bins},{y1_bins}), stakes=({y0_stakes},{y1_stakes})."

    ''' --------
    ## reproject data to SWISS_CRS for evaluation
    ------------'''
    
    ## reprojecting dataset can only be for 2D or 3D arras, so need to map/loop one dimension of N/F
    tmp_ds_list = []
    for n_value in ds_glacier_mb.Nlength.values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
            ds_tmp_n = ds_glacier_mb.sel(Nlength=n_value).transpose('Fparam','y','x')
            tmp_ds_list.append(ds_tmp_n.rio.reproject(swiss_crs))
    ds_glacier_mb = xr.concat(tmp_ds_list, dim='Nlength') ## (Nlength, Fparam, y, x) in swiss_crs

    ## reproject f111:
    tmp_ds_list_f111_Ng = []
    for ng_value in ds_glacier_mb_f111.Ngrad.values:
        tmp_ds_list_f111_Nf = []
        for nf_value in ds_glacier_mb_f111.Nfdiv.values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
                ds_tmp_nf_ng = ds_glacier_mb_f111.sel(Ngrad=ng_value, Nfdiv=nf_value).transpose('Fparam','y','x')
                tmp_ds_list_f111_Nf.append(ds_tmp_nf_ng.rio.reproject(swiss_crs))
        tmp_ds_list_f111_Ng.append(xr.concat(tmp_ds_list_f111_Nf, dim='Nfdiv'))
    ds_glacier_mb_f111 = xr.concat(tmp_ds_list_f111_Ng, dim='Ngrad') ## (Ng, Nlength, Fparam, y, x) in swiss_crs

    pd_stake_data_yyyy_avg, pd_stake_data_incl_weights = evalF.stake_data_get_weights_and_summary(pd_stake_data_yyyy)

    ds_wrmse_stakes  = evalF.calculate_weighted_rmse_at_stakes_all_fXXX(
                                    ds_glacier_mb, ds_glacier_mb_f111, 
                                    pd_stake_data_yyyy_avg, pd_stake_data_incl_weights )
    # ds_wrmse_stakes is xr.dataset (Nlength, Fparam, Ngrad, Nfdiv) with variables rmse_f000, rmse_f001, rmse_f110, rmse_f111

    

    '''#############################################################
    Assembling output
    -- values are still for one specific dhdt period
    ################################################################'''
    
    ds_rmse_elevbins.attrs['glacier_sgiid'] = glacier_sgiid
    ds_rmse_elevbins.attrs['glacier_rgiid'] = glacier_rgiid
    ds_rmse_elevbins.attrs['glacier_name'] = glacier_name
    ds_rmse_elevbins.attrs['dhdt_period'] = dhdt_period
    ds_rmse_elevbins.attrs['used elev_bins hmin'] = str(hmin_binned.tolist())
    ds_rmse_elevbins.attrs['y0_y1 glamos data bins'] = f"{y0_bins}-{y1_bins}"
    ds_rmse_elevbins.attrs['y0_y1 glamos data stakes'] = f"{y0_stakes}-{y1_stakes}"
    ds_rmse_elevbins.attrs['number of stake measurements'] = len(pd_stake_data_yyyy)

    if save_nc:
        # save to nc file
        if not os.path.isdir(os.path.join(path2saveRSME,f'dhdt_{dhdt_period}')):
            os.makedirs(os.path.join(path2saveRSME,f'dhdt_{dhdt_period}'),exist_ok=True)
        
        ## save elevbins nc
        file_elevbins = os.path.join(path2saveRSME,f'dhdt_{dhdt_period}',f'{glacier_sgiid}_{glacier_rgiid}_{glacier_name}_{dhdt_period}_rmse_glamos_elevbins.nc') 
        if not os.path.exists(file_elevbins):
             print('.. saving ds_rmse_elevbins')
             ds_rmse_elevbins.to_netcdf(file_elevbins)
        else:
            print(f"File {os.path.basename(file_elevbins)} already exists. Skipping saving ds_rmse_elevbins.")
        
        ## save stakes nc
        file_stakes = os.path.join(path2saveRSME,f'dhdt_{dhdt_period}',f'{glacier_sgiid}_{glacier_rgiid}_{glacier_name}_{dhdt_period}_weighted-rmse_glamos_stakes.nc')
        # ds_wrmse_stakes.to_netcdf(file_stakes)
        if not os.path.exists(file_stakes):
             print('.. saving weighted ds_wrmse_stakes')
             ds_wrmse_stakes.to_netcdf(file_stakes)
        else:
            print(f"File {os.path.basename(file_stakes)} already exists. Skipping saving ds_rmse_stakes.")
        
    # del gdf_glacier_rgi, glacier_sgiid
    
print('DONE')

# %%
