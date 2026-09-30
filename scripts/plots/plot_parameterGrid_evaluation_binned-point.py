#%%
import xarray as xr
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns 
import ast

import warnings

# Add the directory containing functions.py to path
import sys
sys.path.append(os.path.dirname(os.path.abspath('../scripts')))
import myFunctions as myf
import evalFunctions as evalF

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
# data_dir = '../data'

path2glacier_output = os.path.join('../../data/bruteForce/')
path2glacier_output = os.path.join(data_dir,'bruteForceTMP/glaciers/')


path2save_figure = '../../figures/plot_evaluation_parameterGrid/'
os.makedirs(path2save_figure, exist_ok=True)


#%%

'''------------------
## Glacier outlines (RGI shapefiles)
------------------'''

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs)


''' ################################################################
Grab thickness & elevation data
#################################################################### '''

## load preprocessed reprojected file

file_grab_thickness = os.path.join(data_dir,'SwissGlacierThickness-R2020/04_IceThickness_SwissAlps/IceThickness_50m_epsg32632.tif')
da_grab = xr.open_dataarray(file_grab_thickness).isel(band=0)
da_grab.rio.crs

file_grab_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/IceElevation_50m_epsg32632.tif') 
da_grab_elev = xr.open_dataarray(file_grab_elevation).isel(band=0)
da_grab_elev.rio.crs # EPSG 32632


''' ################################################################
Load GLAMOS data
#################################################################### '''


glamos_traintest_file = os.path.join('../../files/glamos_train_test_split.csv')
glamos_traintest = pd.read_csv(glamos_traintest_file, index_col=0)
glamos_traintest['RGI_match'] = glamos_traintest['RGI_match'].apply(ast.literal_eval)
glamos_traintest.rename(columns={'name':'glacier_name'}, inplace=True)

df_glamos_sgi = glamos_traintest.copy()
df_glamos_sgi.reset_index(inplace=True) ## SGIID as column



#%%
''' ################################################################
Load point-data from GLAMOS stakes for evaluation
- in CH1903 coordinate system; == EPSG 2056 (opgvolger van EPSG 21781, but glamos still in 21781)
#################################################################### '''

## organise filnames that have simplified SGI name / to match glacier sgi-IDs 
glamos_stakes_filelist = os.listdir(os.path.join(data_dir,'GLAMOS/massbalance_point_2021_r2021/annual/'))
glamos_stakes_filelist.sort()

## open table with matched filenames
glamos_names_matched = pd.read_csv('../../files/glamos_sgi_name-matches.csv')
glamos_names_matched


## elevation-binned file
path2data_glamos = os.path.join(data_dir,'GLAMOS/massbalance_fixdate_elevationbins.csv')
df_glamos_mb = myf.load_glamos_elevationbins(path2data_glamos)




#%% 
''' ################################################################
Make plots for ALL GLACIERs in GLAMOS dataset
#################################################################### '''

## all parameter options
N_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #  ds_glacier_smb['Nlength'].values.tolist()
F_values = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0] # ds_glacier_smb['Fparam'].values.tolist()
Ng_values = [0, 1, 2, 3, 4, 5] # ds_glacier_smb_f111['Ngrad'].values.tolist()
Nf_values =[0, 1, 2, 3, 4, 5]  # ds_glacier_smb_f111['Nfdiv'].values.tolist()
   
best_param_dict = pd.read_json('../../files/best_parameters_bruteforce_weighted.json')

N000 = 0; F000 = 0.75
N001 = best_param_dict['f001']['N']; 
F001 = best_param_dict['f001']['F']
N110 = best_param_dict['f110']['N']; 
F110 = best_param_dict['f110']['F']
Ng111 = best_param_dict['f111']['Ng']; 
Nf111 = best_param_dict['f111']['Nf']; 
F111 = best_param_dict['f111']['F']


## save toggle
save_fig_stakes = False
save_fig_elevbin = False

#%%
''' ################################################################
Loop GLAMOS glaciers
#################################################################### '''

df_rmse_list = []
for glacier_sgiid in ['B36-26', 'B56-03']: # Aletsch, Findel  //  glamos_traintest.index:

    traintest_subset = glamos_traintest.loc[glamos_traintest.index == glacier_sgiid]['set'].item()

    '''## select a single glacier to evaluate'''
    glacier_stakefile = glamos_names_matched.loc[glamos_names_matched['SGIID']==glacier_sgiid]['stake_csv'].item()
    glacier_stakefiles = []
    assert isinstance(glacier_stakefile, str), f"Glacier stakefile for {glacier_sgiid} is not a string."


    ## glacier gdf
    gdf_glacier_sgi = df_glamos_sgi.loc[df_glamos_sgi['sgi-id']== glacier_sgiid].copy() # select a row with multiple RGI  matches
    glacier_sgiid = gdf_glacier_sgi['sgi-id'].item()
    glacier_name = gdf_glacier_sgi['glacier_name'].item()
    print('-----')
    print(glacier_sgiid, glacier_name , ';', glacier_stakefile)

    '''--------------------
    ## Find the RGI-id match
    Some sgi-glaciers have multiple rgi matches, because the RGI idx identifies some tributary glaciers seperately while the SGI idx counts them as one glacier.
    But I've only calculated fluxdiv/smb for RGI glaciers >2km. So some of these RGI-glaciers that are together in one SGI-glacier might not have been processed.
    --------------------'''

    glacier_rgiid_list = gdf_glacier_sgi['RGI_match'].item() 
    
    ## get corresponding RGI
    glacier_rgiid = df_glamos_sgi.loc[df_glamos_sgi['sgi-id']==glacier_sgiid ]['RGI_match'].values[0]
    if glacier_sgiid == 'A50i-19': ## manual fix for Clairdenfirn, as i've merged the 4 RGIs it consists of
        glacier_rgiid = 'RGI60-11.008merged'
    else:
        glacier_rgiid = glacier_rgiid[0]
    gdf_glacier_rgi = gdf_swiss_large.loc[gdf_swiss_large['RGIId'] == glacier_rgiid].copy()
    
    ## Extract glacier elevation data and the RGI outlines
    da_elev = da_grab_elev.rio.clip( gdf_glacier_rgi.geometry, drop=True,all_touched=True) ## still in EPSG 32632
    da_thickness = da_grab.rio.clip( gdf_glacier_rgi.geometry, drop=True,all_touched=True)
    
    gdf_glacier_rgi.to_crs(swiss_crs ,inplace=True) ## to EPSG:21781
    
    
    ''' ################################################################
    Load output data from brute-force runs
    #################################################################### '''

    path2glacier = os.path.join(path2glacier_output, glacier_rgiid)
    filen_nc_fdiv = os.path.join(path2glacier, f'{glacier_rgiid}_fluxdiv_paramgrid.nc' )

    
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
    ds_glacier_mb_f111.rio.write_crs('EPSG:32632', inplace=True) # (y,x, Nf, Ng, Fparam)

    
    ''' --------------------
    # #### get stake data of glacier
    # mb_we is in [mm w.e. / kg m-2]	; my values are in m w.e. , so divide by 1000
    # mb_error :Uncertainty of point mass balance as square root of the sum of squares of the fractional uncertainties of Density and Raw Balance [mm w.e.]) --> also divide by 1000
    --------------------'''

    pd_stake_data_yyyy, y0_stakes, y1_stakes = evalF.load_stake_data(
                            os.path.join(data_dir,'GLAMOS/massbalance_point_2021_r2021/annual_preprocessed/',glacier_stakefile),
                            start_year=glamos_y0,
    )
    
    
    ''' ################################
    # Bin 2D-MB to elevation
    ################################### '''

    '''## GLAMOS Evaluation mass balance data'''
    df_glamos_yyyy = df_glamos_mb.loc[ df_glamos_mb['date_start'].dt.year > glamos_y0].copy()
    df_mb_glacier, glac_name_ID = myf.get_glacier_mb_bins_glamos(df_glamos_yyyy, 
                                            glacier_id=glacier_sgiid)
    hmin_bin_glamos = df_mb_glacier.index.to_list() # x-axis values: glamos is 100m bin step

    ''' --------------------
    ## Bin my own calculated SMB using Grab. elevation bins
    ds_glacier_smb is (N,F,y,x)
    smb_bin_means_f000 is (N, F, h_bin)
    -------------------- '''

    # smb_bin_means, hmin_binned, bin_std, bin_sem  = myf.bin_da_to_elevbins(da_smb, da_elev)
    mb_bin_means_f000, hmin_binned, bin_std_f000, mb_bin_sem_f000  = myf.bin_da_to_elevbins(ds_glacier_mb['mb_f000'], da_elev, binstep=100)
    mb_bin_means_f001, hmin_binned, bin_std_f001, mb_bin_sem_f001 = myf.bin_da_to_elevbins(ds_glacier_mb['mb_f001'], da_elev, binstep=100)
    mb_bin_means_f110, hmin_binned, bin_std_f110, mb_bin_sem_f110 = myf.bin_da_to_elevbins(ds_glacier_mb['mb_f110'], da_elev, binstep=100)
    mb_bin_means_f111, hmin_binned, bin_std_f111, mb_bin_sem_f111 = myf.bin_da_to_elevbins(ds_glacier_mb_f111['mb_f111'], da_elev, binstep=100)

    # Stack all three datasets
    mb_datasets = {
        'MB: no smoothing (f000)': mb_bin_means_f000,
        'MB: smooth only f.div. (f001)': mb_bin_means_f001, 
        'MB: smooth only gradients (f110)': mb_bin_means_f110,
        'MB: smooth both (f111)': mb_bin_means_f111
    }
    mb_sem_datasets = {
        'MB: no smoothing (f000)': mb_bin_sem_f000,
        'MB: smooth only f.div. (f001)': mb_bin_sem_f001, 
        'MB: smooth only gradients (f110)': mb_bin_sem_f110,
        'MB: smooth both (f111)': mb_bin_sem_f111
    }
    mb_std_datasets = {
        'MB: no smoothing (f000)': bin_std_f000,
        'MB: smooth only f.div. (f001)': bin_std_f001, 
        'MB: smooth only gradients (f110)': bin_std_f110,
        'MB: smooth both (f111)': bin_std_f111
    }


    ''' --------------------
    get best parameter info 
    ------------------------'''

    mb_bestparams_dataset = {
        'mb_f000':   mb_bin_means_f000[N_values.index(N000), F_values.index(F000), :], ## mb_bin_means_f000 is shape (N,F,h_bin)
        'mb_f001':   mb_bin_means_f001[N_values.index(N001), F_values.index(F001), :], 
        'mb_f110':   mb_bin_means_f110[N_values.index(N110), F_values.index(F110), :],
        'mb_f111':   mb_bin_means_f111[Nf_values.index(Nf111),Ng_values.index(Ng111), F_values.index(F111), :]
    }
    ## select choice of bestparams
    df_bestparams = pd.DataFrame(mb_bestparams_dataset)
    df_bestparams.index = hmin_binned ## add hmin_binned as index to the dataframe

    
    ''' --------------------
    ### Plot binned eval
    ------------------------'''
    
    plot_err = None
    fig,ax = evalF.plot_smb_elevbins_with_stakes(df_mb_glacier, 
                                    mb_datasets, df_bestparams,
                                    mb_sem_datasets, plot_err,
                                    hmin_binned,
                                    )
    ax.set_title(f'{glacier_sgiid}, {glacier_rgiid}, {glacier_name}')
    if glacier_sgiid == 'B22-01': # glacier tsanfleuron, reset ylim
        ylim_new = [-10, 5]
        ax.set_ylim(ylim_new)
    if glacier_sgiid == 'B43-03': # glacier Rhone, reset ylim
        ylim_new = [-25, 10]
        ax.set_ylim(ylim_new)

    ## get legend and update legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['GLAMOS Min-Max',
                    'GLAMOS Mean',
                    'No filter (f000)',
                    'Filter flux div. (f001)',
                    'Filter gradients (f110)',
                    'Filter both (f111)']
    ax.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    if save_fig_elevbin:
        filename = f'{glacier_sgiid}_{glacier_rgiid}_{glacier_name}_{dhdt_period}_glamos-binned-stakes_paramgrid.pdf'
        os.makedirs(path2save_figure, exist_ok=True)
        fig.savefig(os.path.join(path2save_figure,filename ), dpi=300, bbox_inches='tight')
    # plt.close(fig)
    
    ''' --------
    ## get RMSE at stake locations
    Include a buffer around the stake point, for uncertainty estimation 
    --> exact stake location + std around stake (3x3 px window around stake)

    +reproject data to SWISS_CRS for evaluation
    ------------'''
    
    '''## Reproject to swiss coords'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
        da_mb_f000 = ds_glacier_mb['mb_f000'].sel(Nlength=N000, Fparam=F000).rio.reproject(swiss_crs)
        da_mb_f001 = ds_glacier_mb['mb_f001'].sel(Nlength=N001, Fparam=F001).rio.reproject(swiss_crs)
        da_mb_f110 = ds_glacier_mb['mb_f110'].sel(Nlength=N110, Fparam=F110,method='nearest').rio.reproject(swiss_crs)
        da_mb_f111 = ds_glacier_mb_f111['mb_f111'].sel(
                            Ngrad=Ng111, Nfdiv=Nf111, Fparam=F111, method='nearest'
                            ).rio.reproject(swiss_crs)

    ## summary of stake values (observaionts)
    pd_stake_data_yyyy_avg, _ = evalF.stake_data_get_weights_and_summary(pd_stake_data_yyyy )

    ''' ## extract my MB at stake location, incl buffer '''
    pd_stake_data_derivedMB = evalF.sample_mb_at_stake_locations(
                                {'f000': da_mb_f000,
                                 'f001': da_mb_f001,
                                 'f110': da_mb_f110,
                                 'f111': da_mb_f111},
                                pd_stake_data_yyyy,
                                buffer_size_N=1.5,
                                resolution_m=50)

    ## add summary of derived MB values; unstack multi-columns
    pd_mb_fXXX_summary = pd_stake_data_derivedMB.groupby(
        'matched_stakeID', observed=False).agg(
        {   **{f'mb_{fXXX}_point': 'mean'       for fXXX in ('f000', 'f001', 'f110', 'f111') },
            **{f'mb_{fXXX}_buffer_mean': 'mean' for fXXX in ('f000', 'f001', 'f110', 'f111') },
            **{f'mb_{fXXX}_buffer_std': 'mean'  for fXXX in ('f000', 'f001', 'f110', 'f111') },
        }
    ).reset_index()

    ## merge to exisitng dataframe
    pd_stake_data_yyyy_avg = pd_stake_data_yyyy_avg.merge(
        pd_mb_fXXX_summary, on='matched_stakeID', how='left'
    )
    

    pd_stake_data_yyyy_avg['date0_yr'] = pd_stake_data_yyyy_avg['date0_dt'].dt.year
    pd_stake_data_yyyy_avg['delta_yr'] = pd_stake_data_yyyy_avg['date1_dt'].dt.year - pd_stake_data_yyyy_avg['date0_yr']
    ## make new column that merges mb_we_std (temporal stdev of multiple stake readings) with mb_error (incase of 1 reading, use reading-error)
    ## fill 'mb_uncertainty' with mb_we_std as priority, otherwise mb_error.

    pd_stake_data_yyyy_avg['mb_uncertainty'] = pd_stake_data_yyyy_avg['mb_we_std'].combine_first(pd_stake_data_yyyy_avg['mb_error'])
    ## can also use PCT range as uncertainty. For stakes with n=1 , again use mb_error
    pd_stake_data_yyyy_avg['mb_pct_range'] = pd_stake_data_yyyy_avg['mb_we_pct95'] - pd_stake_data_yyyy_avg['mb_we_pct05']
    pd_stake_data_yyyy_avg['mb_pct_range'] = pd_stake_data_yyyy_avg['mb_pct_range'].where(pd_stake_data_yyyy_avg['mb_we_count']>1, pd_stake_data_yyyy_avg['mb_error'])


    ''' ## Calculate weighted RMSE for how many readings a stake has'''
    
    rmse_values = {}
    for fXXX in ['f000', 'f001', 'f110','f111']:
        rmse_values[fXXX] = evalF.weighted_rmse(pd_stake_data_yyyy_avg['mb_we_mean'], #obs
                               pd_stake_data_yyyy_avg[f'mb_{fXXX}_point'], #predictions
                                pd_stake_data_yyyy_avg['mb_we_count'] ## weigh by number of readings per stake, so that stakes with more readings (and thus more reliable average) have more weight in the RMSE calculation
                                )
    
    '''-------------------
    ## plot stake data :  scatter
    ----------------------'''
    ## rename columns
    pd_stake_data_yyyy_avg = pd_stake_data_yyyy_avg.rename(
        columns={  # **{f'mb_{fXXX}_buffer_mean': f'mb_{fXXX}'      for fXXX in ('f000', 'f001', 'f110', 'f111') },
                    **{f'mb_{fXXX}_point': f'mb_{fXXX}' for fXXX in ('f000', 'f001', 'f110', 'f111') },
        }
    )
    
    '''## Scatterplot with RMSE values: weighted rmse and with f111'''
    # number of years in current dhdt period (for color pallete)
    y0, y1  = [int(x) for x in dhdt_period.split('-')]
    dy=y1-y0
    
    
    ## add empty rows with delta_yr ranging from 1 to 20, to have same color range for all glaciers even if some glaciers don't have stakes with the same dy
    for delta_yr in range(1, dy+1):
        if delta_yr not in pd_stake_data_yyyy_avg['delta_yr'].values:
            empty_row = pd.DataFrame({
                'matched_stakeID': [np.nan], 
                'mb_we_mean': [np.nan], 'mb_we_min': [np.nan], 'mb_we_max': [np.nan],
                'mb_we_count': [0],
                'mb_we_std': [np.nan], 'mb_we_pct05': [np.nan],'mb_we_pct95': [np.nan],
                'mb_f000': [np.nan],'mb_f001': [np.nan],'mb_f110': [np.nan],'mb_f111': [np.nan],
                'mb_f000_buffer_std': [np.nan],'mb_f001_buffer_std': [np.nan],'mb_f110_buffer_std': [np.nan],'mb_f111_buffer_std': [np.nan],
                'date0_dt': [pd.to_datetime('2000-01-01')],  # dummy date
                'date1_dt': [pd.to_datetime('2000-01-01')],  # dummy date
                'date0_yr': [2000],  # dummy year
                'delta_yr': [delta_yr]  # add missing delta_yr value
            })
            pd_stake_data_yyyy_avg = pd.concat([pd_stake_data_yyyy_avg, empty_row], ignore_index=True)

    
    # fig,axs = plt.subplots(1,3)
    # fig, axs = plt.subplots(1,3,figsize=(17,6), sharex=True )# sharey=True)
    # fig, axs = plt.subplots(1,4,figsize=(22,6), sharex=True )# sharey=True)
    fig, axs = plt.subplots(1,4,figsize=(18,6), sharex=True )# sharey=True)
    # plt.rcParams.update({'font.size': 14})

    # horiz_errorvar = 'mb_uncertainty'
    horiz_errorvar = 'mb_pct_range'
    markersize = 200 # 100, 80
    
    # cmap_nr_obs = sns.color_palette("icefire" ,n_colors=24 )[:20]; norm = plt.Normalize(1,24)
    # cmap_nr_obs = sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
    cmap_nr_obs = sns.color_palette("cubehelix_r" ,n_colors=40 )[10:30]; norm=plt.Normalize(1,20)

    # f000
    ax = axs[0] ## F000 is allowed to have different xlim and ylim, as it is the worst performing model
    _ , ax = evalF.plot_stake_smb_scatter_ax(pd_stake_data_yyyy_avg, 'f000', 
                                xvarname='mb_we_mean',errname=horiz_errorvar,
                                N_F_str = f'N={N000}, F={F000}',
                                hue_var='delta_yr',  msize=markersize, edgecolor='k',
                                style_var=None, #'delta_yr',
                                plot_palette=cmap_nr_obs,
                                rmse_value=rmse_values['f000'], add_legend=False,
                                ax=ax)
    # f001
    ax = axs[1]
    _ , ax = evalF.plot_stake_smb_scatter_ax(pd_stake_data_yyyy_avg, 'f001', 
                                xvarname='mb_we_mean',errname=horiz_errorvar,
                                N_F_str = f'N={N001}, F={F001}',
                                hue_var='delta_yr', msize=markersize,
                                style_var=None,edgecolor='k', # 'k',
                                plot_palette=cmap_nr_obs,
                                rmse_value=rmse_values['f001'], add_legend=False,
                                ax=ax)
    xlim_f001 = ax.get_xlim(); ylim_f001 = ax.get_ylim() ## get xlim and ylim of f001 plot to set same limits for f000 and f110 plots

    # f110
    ax = axs[2]
    _ , ax = evalF.plot_stake_smb_scatter_ax(pd_stake_data_yyyy_avg, 'f110', 
                                xvarname='mb_we_mean',errname=horiz_errorvar,
                                N_F_str = f'N={N110}, F={F110}', 
                                style_var=None, #'delta_yr',
                                msize=markersize,edgecolor='k', # 'k',
                                hue_var='delta_yr', plot_palette=cmap_nr_obs,
                                rmse_value=rmse_values['f110'], add_legend=False,
                                ax=ax)
    ax.set_xlim(xlim_f001) ## set same xlim for f001, f110, f111 subplots
    ax.set_ylim(ylim_f001) 
    

    ax = axs[3]
    _ , ax = evalF.plot_stake_smb_scatter_ax(pd_stake_data_yyyy_avg, 'f111', #glacier_sgiid, glacier_name, #dhdt_period, 
                                xvarname='mb_we_mean',errname=horiz_errorvar,
                                N_F_str = f'Ng={Ng111}, Nf={Nf111}, F={F111}', 
                                style_var=None, #'delta_yr',
                                msize=markersize, edgecolor='k', # 'k',
                                hue_var='delta_yr', plot_palette=cmap_nr_obs,
                                rmse_value=rmse_values['f111'], add_legend=False,
                                ax=ax)
    ax.set_xlim(xlim_f001) ## set same xlim for f001, f110, f111 subplots
    ax.set_ylim(ylim_f001) 
    
    fig.tight_layout()

    '''## add colorbar of delta_yr'''
    cbar_width= 0.005#25
    
    import matplotlib as mpl
    cmap_add = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', cmap_nr_obs, N=20)
    # sm = plt.cm.ScalarMappable(cmap='icefire', norm=norm)
    sm = plt.cm.ScalarMappable(cmap=cmap_add, norm=norm)
    
    ## create axes for cbar and add it
    sm.set_array([])
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0 + 0.1, cbar_width, ax.get_position().height / 1.5])
    ax.figure.colorbar(sm, cax=cax, label='Number of observations', 
                        # ticks=range(1,21,5),
                        ticks=[1,5,10,15,20],
                        ) # add colorbar with ticks every 2 years    
    
    if save_fig_stakes:
        # filename = f'{glacier_sgiid}_{glacier_rgiid}_{glacier_name}_{dhdt_period}_scatter_glamos-stakes_bestParam.jpg'
        filename = f'{glacier_sgiid}_{glacier_rgiid}_{glacier_name}_{dhdt_period}_scatter_glamos-stakes_bestParam_wRMSE.pdf'
        fig.savefig(os.path.join(path2save_figure,filename ), dpi=300, bbox_inches='tight')
    # plt.close(fig)

    
# %%
