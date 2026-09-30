#%%
import glob

import xarray as xr
# import rasterio as rio
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import seaborn as sns 
import ast

import warnings

import sys
sys.path.append(os.path.dirname(os.path.abspath('../scripts')))
import myFunctions as myf
import evalFunctions as evalF

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
path2mb ='../../data/bestParams'

path2save_figure = '../../figures/plot_performance/'

my_palette = ['#2b6f39','#efbb1a','#d490c6'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#



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

file_grab_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/IceElevation_50m_epsg32632.tif') 
da_grab_elev = xr.open_dataarray(file_grab_elevation).isel(band=0)


#%%

''' ################################################################
Load GLAMOS data
#################################################################### '''

glamos_traintest_file = os.path.join('../../files/glamos_train_test_split.csv')
glamos_traintest = pd.read_csv(glamos_traintest_file, index_col=0)
glamos_traintest['RGI_match'] = glamos_traintest['RGI_match'].apply(ast.literal_eval)
df_glamos_sgi = glamos_traintest.copy()
glamos_traintest.rename(columns={'name':'glacier_name'}, inplace=True)
df_glamos_sgi.reset_index(inplace=True) ## SGIID as column


#%%
''' ################################################################
Load point-data from GLAMOS stakes for evaluation
- in CH1903 coordinate system; == EPSG 2056 (opgvolger van EPSG 21781, but glamos still in 21781)
#################################################################### '''

## organise filnames that have simplified SGI name / to match glacier sgi-IDs 
glamos_stakes_filelist = os.listdir(os.path.join(data_dir,'GLAMOS/massbalance_point_2021_r2021/annual/'))
glamos_stakes_filelist.sort()


## elevation-binned file
path2data_glamos = os.path.join(data_dir,'GLAMOS/massbalance_fixdate_elevationbins.csv')
df_glamos_mb = myf.load_glamos_elevationbins(path2data_glamos)


## open table with matched filenames
glamos_names_matched = pd.read_csv('../../files/glamos_sgi_name-matches.csv')




#%% 
''' ################################################################
Make plots for ALL GLACIERs in GLAMOS dataset
#################################################################### '''

''' --------------------
get best parameter info 
------------------------'''

## bestParams choice
best_param_dict = pd.read_json('../../files/best_parameters_bruteforce_weighted.json')

best_approach = 'f001' # update this to select which approach to plot as thick line in the end
best_approach_label = 'SMB: smooth only f.div. (f001)'
best_N = best_param_dict['f001']['N'];  
best_F = best_param_dict['f001']['F']; 
best_F_str = f'{best_F:.2f}'.replace('.','') ## for filename only

''' --------------------
######  select which timeperiod for calculations
--------------------'''

glamos_y0 = 2000; dhdt_period = '2000-2020'
# glamos_y0 = 2010; dhdt_period = '2010-2020'
# glamos_y0 = 2015; dhdt_period = '2015-2020'

''' --------
## get list of output files
# ---------- '''

# path2bestoutput = path2smb # os.path.join(path2smb,f'mb_{dhdt_period}')
fnames_mb_list = glob.glob(os.path.join(path2mb,f'mb_{dhdt_period}/', '*.tif' ))

''' ## saving toggle'''
save_fig = False

''' ################################################################
Loop GLAMOS glaciers
#################################################################### '''
list_df_glacier_values = []

for glacier_sgiid in glamos_traintest.index[[0, 9]]: ## Aletsch & Findel
    traintest_subset = glamos_traintest.loc[glamos_traintest.index == glacier_sgiid]['set'].item()

    '''## select a single glacier to evaluate'''
    # glacier_stakefile = glamos_names_matched.loc[glamos_names_matched['SGIID']==glacier_sgiid]['stake_file'].item()
    # glacier_stakefiles = [f for f in glacier_stakefile.split('; ')]
    glacier_stakefile = glamos_names_matched.loc[glamos_names_matched['SGIID']==glacier_sgiid]['stake_csv'].item()
    glacier_stakefiles = []
    assert isinstance(glacier_stakefile, str), f"Glacier stakefile for {glacier_sgiid} is not a string."


    ## glacier gdf
    df_glacier_sgi = df_glamos_sgi.loc[df_glamos_sgi['sgi-id']== glacier_sgiid].copy() # select a row with multiple RGI  matches
    glacier_sgiid = df_glacier_sgi['sgi-id'].item()
    glacier_name = df_glacier_sgi['name'].item()
    print('-----')
    print(glacier_sgiid, glacier_name , ';', glacier_stakefile)

    '''--------------------
    ## Find the RGI-id match
    Some sgi-glaciers have multiple rgi matches, because the RGI idx identifies some tributary glaciers seperately while the SGI idx counts them as one glacier.
    But I've only calculated fluxdiv/smb for RGI glaciers >2km. So some of these RGI-glaciers that are together in one SGI-glacier might not have been processed.
    --------------------'''

    glacier_rgiid_list = df_glacier_sgi['RGI_match'].item() 
    
    ## get corresponding RGI
    glacier_rgiid = df_glamos_sgi.loc[df_glamos_sgi['sgi-id']==glacier_sgiid ]['RGI_match'].values[0]
    if len(glacier_rgiid) >1:
        # if glacier_sgiid == 'A50i-19':
        #     ## manual fix for Clairdenfirn, as i've merged the 4 RGIs it consists of
        glacier_rgiid = 'RGI60-11.008merged'

    glacier_rgiid = glacier_rgiid[0]
    gdf_glacier_rgi = gdf_swiss_large.loc[gdf_swiss_large['RGIId'] == glacier_rgiid].copy()

    
    ## Extract glacier elevation data and the RGI outlines
    da_elev = da_grab_elev.rio.clip( gdf_glacier_rgi.geometry, drop=True,all_touched=True) ## still in EPSG 32632
    
    gdf_glacier_rgi.to_crs(swiss_crs ,inplace=True) ## to EPSG:21781
    
    
    ''' --------------------
    # #### get stake data of glacier
    # mb_we is in [mm w.e. / kg m-2]	; my values are in m w.e. , so divide by 1000
    # mb_error :Uncertainty of point mass balance as square root of the sum of squares of the fractional uncertainties of Density and Raw Balance [mm w.e.]) --> also divide by 1000
    --------------------'''

    pd_stake_data_yyyy, y0_stakes, y1_stakes = evalF.load_stake_data(
                            os.path.join(data_dir,'GLAMOS/massbalance_point_2021_r2021/annual_preprocessed/',glacier_stakefile),
                            start_year=glamos_y0,
    )
    


    ''' ################################################################
    Load output data from brute-force runs
    --> will need to make a 'best choice' selection of parameter space
    --> only evaluate 'best choice' to stakes? or all?
    #################################################################### '''

    ''' --------------------
    load corresponding mb
    ------------------------'''

    fname_mb_rgiid = [os.path.basename(f) for f in fnames_mb_list if glacier_rgiid in f]
    assert len(fname_mb_rgiid) == 1, f'Not exactly 1 file found for glacier {glacier_rgiid}'
    da_mb_bestparams = xr.open_dataarray(
                    os.path.join(path2mb,f'mb_{dhdt_period}/', fname_mb_rgiid[0])
                    ).isel(band=0).drop_vars('band').rename('mb') ## epsg 32632
    da_mb_bestparams.attrs['units'] = 'm w.e. yr-1'
    da_mb_bestparams.attrs['long_name'] = 'mass balance'
    
    

    
    ''' ################################
    # Bin 2D-MB to elevation
    ################################### '''

    '''## GLAMOS Evaluation mass balance data'''
    df_glamos_yyyy = df_glamos_mb.loc[ (df_glamos_mb['date_start'].dt.year >= glamos_y0) & \
                                       (df_glamos_mb['date_start'].dt.year <= 2020) ].copy() # only data that is in same dhdt period
    df_mb_glacier, glac_name_ID = myf.get_glacier_mb_bins_glamos(df_glamos_yyyy, 
                                            glacier_id=glacier_sgiid)
    hmin_bin_glamos = df_mb_glacier.index.to_list() # x-axis values: glamos is 100m bin step
     
    # GLAMOS: mm.we to m.w.e.
    df_mb_Ba = (df_mb_glacier[['Ba']]/1000).copy()
    df_mb_glacier = pd.merge(df_mb_glacier[['date_start','area','h_max']],df_mb_Ba, on='h_min', how='left')

    

    ''' ---- bin MB to elevation ------- '''
    ## clip to glacier outline
    # da_mb_bestparams = da_mb_bestparams.rio.clip(gdf_glacier_rgi.to_crs(da_mb_bestparams.rio.crs).geometry, drop=False,all_touched=True)

    mb_bin_means_bestparams, hmin_binned, bin_std, mb_bin_sem = myf.bin_da_to_elevbins(da_mb_bestparams, da_elev, binstep=100)

    ## For claridenfirn, remove pixel below ice fall, < 2500 m a.s.l. 
    if glacier_sgiid == 'A50i-19': 
        '''The glacier used to terminate in an ice fall with lowest elevation at 2342 m a.s.l. (in 2010, WGMS, 2026), 
        but in the past decade(s) the terminus has retreated up to 2534 in 2017 and 2567 m a.s.l. in 2022 (WGMSlatest).
        '''
        hbin_threshold = 2500
        # Filter out bins below the threshold: my data
        valid_bins_mask = hmin_binned >= hbin_threshold
        mb_bin_means_bestparams = mb_bin_means_bestparams[valid_bins_mask]
        hmin_binned = hmin_binned[valid_bins_mask]
        bin_std     = bin_std[valid_bins_mask]
        mb_bin_sem = mb_bin_sem[valid_bins_mask]
        # also filter out bins of GLAMOS data
        df_mb_glacier = df_mb_glacier[df_mb_glacier.index >= hbin_threshold]

    df_bestparams = pd.DataFrame([hmin_binned, mb_bin_means_bestparams, bin_std]).T 
    df_bestparams.columns = ['hmin', 'mb', 'std']
    df_bestparams.index = hmin_binned
    
    '''--------------------
    Calculate RMSE wrt elevation-bins
    -------------------- '''
    idx_to_select_data = np.isin(hmin_binned, df_mb_glacier.index.values)
    idx_to_select_glamos = np.isin(df_mb_glacier.index.values, hmin_binned)
    rmse_bins_bestparams = np.sqrt(np.nanmean((mb_bin_means_bestparams[idx_to_select_data] - df_mb_glacier[('Ba', 'mean')].values[idx_to_select_glamos]) ** 2, axis=-1))

    
    ''' --------
    ## get RMSE at stake locations
    Map is 50 m resolution, so buffer stakes by 25 m to get mean value around stake location?
    Or do 
    (a) exact stake location and 
    (b) bilinear interpolation of N surrounding pixels?

    --> No do:
    (c) exact stake location + std around stake (3x3 px window around stake)

    +reproject data to SWISS_CRS for evaluation
    ------------'''

    ## reproject to stake CRS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
        da_mb_bestparams = da_mb_bestparams.rio.reproject(swiss_crs)

    '''    # ## get my MB values at the stake locations, incl buffer'''
    pd_stake_data_derivedMB = evalF.sample_mb_at_stake_locations(
                                {best_approach: da_mb_bestparams},
                                pd_stake_data_yyyy,
                                buffer_size_N=1.5,
                                resolution_m=50)

    ## add summary of derived MB values; unstack multicolumns
    pd_mb_fXXX_summary = pd_stake_data_derivedMB.groupby(
        'matched_stakeID', observed=False).agg( {   f'mb_{best_approach}_point': 'mean' ,
                                                    f'mb_{best_approach}_buffer_std': 'mean' } ).reset_index()

    # # ## rename columns
    # pd_mb_fXXX_summary = pd_mb_fXXX_summary.rename(
    #     columns={ #f'mb_{best_approach}_buffer_mean': f'mb_{best_approach}_buffer_mean',
    #               f'mb_{best_approach}_point': f'mb_{best_approach}_point_mean' })

    
    '''## --------
    average unique stakes (some stakes have multiple measurements, so average them to get one value per stake)
    -------------- '''
   
    
    pd_stake_data_yyyy_avg, _ = evalF.stake_data_get_weights_and_summary(pd_stake_data_yyyy)

    pd_stake_data_yyyy_avg['mb_uncertainty'] = pd_stake_data_yyyy_avg['mb_we_std'].combine_first(pd_stake_data_yyyy_avg['mb_error'])
    ## can also use PCT range as uncertainty. For stakes with n=1 , again use mb_error
    pd_stake_data_yyyy_avg['mb_pct_range'] = pd_stake_data_yyyy_avg['mb_we_pct95'] - pd_stake_data_yyyy_avg['mb_we_pct05']
    pd_stake_data_yyyy_avg['mb_pct_range'] = pd_stake_data_yyyy_avg['mb_pct_range'].where(pd_stake_data_yyyy_avg['mb_we_count']>1, pd_stake_data_yyyy_avg['mb_error'])
    pd_stake_data_yyyy_avg['date0_yr'] = pd_stake_data_yyyy_avg['date0_dt'].dt.year
    pd_stake_data_yyyy_avg['delta_yr'] = pd_stake_data_yyyy_avg['date1_dt'].dt.year - pd_stake_data_yyyy_avg['date0_yr']


    ## merge OBS & Derived MB dataframe
    pd_stake_data_yyyy_avg = pd_stake_data_yyyy_avg.merge(
        pd_mb_fXXX_summary, on='matched_stakeID', how='left'
    )
    
    ''' ----------
    ## Calculate weighted RMSE for how many readings a stake has
    -------------- '''
    
    rmse_stakes_bestparams = evalF.weighted_rmse(pd_stake_data_yyyy_avg['mb_we_mean'],
                                pd_stake_data_yyyy_avg[f'mb_{best_approach}_point'], 
                                pd_stake_data_yyyy_avg['mb_we_count'] ## weigh by number of readings per stake, so that stakes with more readings (and thus more reliable average) have more weight in the RMSE calculation
                                )

    ## overall mean RMSE
    rmse_mean = (rmse_bins_bestparams + rmse_stakes_bestparams) / 2

    ## dataframe with RMSE values for current glacier
    df_glacier_values = pd.DataFrame({
        'glacier_sgiid': glacier_sgiid, 'glacier_rgiid': glacier_rgiid, 'glacier_name': glacier_name,
        'traintest': traintest_subset,
        'rmse_bins_bestparams': rmse_bins_bestparams.round(3),
        'rmse_stakes_bestparams': rmse_stakes_bestparams.round(3),
        'rmse_mean': rmse_mean.round(3)
    }, index=[0])
    list_df_glacier_values.append(df_glacier_values)

    
    ''' --------------------
    ### ## PLOT FIGURE
    ------------------------'''

    cmap_smb = mpl.cm.RdBu
    cmap_smb.set_bad("#E2E2E2",1.) # light gray
    plot_rmse_value = True
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 16})

    ##$ update tick fontsize
    fsize_ticklabels = 18
    plt.rcParams.update({'xtick.labelsize': fsize_ticklabels, 'ytick.labelsize': fsize_ticklabels})

    ''' --------------------
    ### ## glacier map
    ------------------------'''

    ## SINGLE FIGURE 3 panels
    ## draft a similar figure size as above but with fixed ax sizes
    w, h = 25, 8
    margin12 = 0.8
    margin = 0.5
    fig = plt.figure(figsize=(w, h), facecolor=None)#'lightblue')
    ax1 = fig.add_axes([(w / 4 + margin) / w,       margin / h, 
                        (w / 4 - 2 * margin12) / w,   (h - 2 * margin) / h
                        ])  # [left, bottom, width, height]
    ax2 = fig.add_axes([(2* w / 4 + margin) / w,    margin / h,      
                        (2*w / 4 - 2 * margin) / w,   (h - 2 * margin) / h
                        ])
    ax3 = fig.add_axes([(4 * w / 4 + margin) / w,   margin / h, 
                        (w / 4 - 2 * margin) / w,   (h - 2 * margin) / h
                        ])

    ax=ax1
    
    # fig,ax=plt.subplots(1, figsize=(6,6) )

    gdf_stake_data_yyyy_avg = pd_stake_data_yyyy.groupby('matched_stakeID', observed=False).agg(
                {'mb_we':'mean','x_pos':'mean','y_pos':'mean'}
         )
    
    gdf_stake_data_yyyy_avg = gpd.GeoDataFrame(
        gdf_stake_data_yyyy_avg,
        geometry=gpd.points_from_xy(gdf_stake_data_yyyy_avg.x_pos, gdf_stake_data_yyyy_avg.y_pos),
        crs='EPSG:21781'
    ) 
    # gdf_stake_data_yyyy_avg.to_crs(da_mb_bestparams.rio.crs, inplace=True) ## reproject to same crs as smb map for plotting
    da_plot_mb = da_mb_bestparams.rio.reproject(gdf_stake_data_yyyy_avg.crs) ## reproject smb map to same crs as stake data for plotting
    
    
    # fig,ax= plt.subplots(figsize=(8,6))
    plot_minmax = [-8,8]
    ax = evalF.plot_glacier_data(da_plot_mb, minmax=plot_minmax,
                                 rgi_outlines=gdf_glacier_rgi, 
                                 cmap=cmap_smb, add_colorbar=True, cbar_vertical=False,
                                 title=f'{glacier_sgiid}, {glacier_name}', clabel='Derived MB (m w.e.)', 
                                    ax=ax, plot_scalebar=True , 
                                 )
    ## add stake points to map
    ax.scatter(gdf_stake_data_yyyy_avg['x_pos'], gdf_stake_data_yyyy_avg['y_pos'],
                    c=gdf_stake_data_yyyy_avg['mb_we'], cmap=cmap_smb, vmin=plot_minmax[0], vmax=plot_minmax[1],
                    s=70, edgecolor='k', label='hoi'
                    )
    ax.set_xlabel('x'); ax.set_ylabel('y')

    
    ''' --------------------
    ### Plot binned eval
    ------------------------'''
    ## select bestparam combo as seperate array for plotting thick line
    color = my_palette[0] # 
    
    
    '''-- lineplot with best-RMSE as thick line '''

    # fig,ax=plt.subplots(1, figsize=(10,6) )
    ax=ax2
    
    ## plot GLAMOS shaded
    h_min_values = df_mb_glacier.index  # x-axis values
    ax.fill_between(h_min_values, df_mb_glacier[('Ba', 'min')], df_mb_glacier[('Ba', 'max')], color='gray', alpha=0.2, label='GLAMOS min/max range')
    ax.plot(h_min_values, df_mb_glacier[('Ba', 'mean')], color='black', linestyle='-.', label='GLAMOS mean')
    
    # Plot MB 
    ax.plot(df_bestparams.index, df_bestparams['mb'], linewidth=3, 
            color=color, linestyle='-', 
            label=f'mean derived MB \n(F001; N={best_N}, F={best_F})', # f'{best_approach_label}; N={best_N}, F={best_F}'
            )
    ## add errorbar for spatial uncertainty
    ax.errorbar(x=df_bestparams.index, y=df_bestparams['mb'], 
                        yerr=df_bestparams['std'], 
                        fmt='none', ecolor=color, alpha=0.7, capsize=3,
                        label='standard-deviation of bin')
    ylim=ax.get_ylim()
    ax.set_xlabel('Minimum elevation of bin [m a.s.l.]')
    ax.set_ylabel('Mass balance [m w.e.]')

    ## line for 0 SMB
    ax.hlines(0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color=[0.1,0.1,0.1], linestyle='--', alpha=0.5)

    ax.legend(loc='lower right',fontsize=fsize_ticklabels+1)
    ax.grid('major', linestyle='--', alpha=0.5)
    
    if plot_rmse_value:
        # ax.text(0.15, 0.05, f'RMSE={rmse_bins_bestparams:.2f}', transform=ax.transAxes, 
        ax.text(0.21, 0.05, f'RMSE={rmse_bins_bestparams:.2f}', transform=ax.transAxes,
                fontsize=fsize_ticklabels,
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle='round', edgecolor='black', facecolor='white',alpha=0.8))
        
    '''-------------------
    ## plot stake data : spatial / scatter
    ----------------------'''

    '''## Scatterplot with RMSE values: weighted rmse and with f111'''
    # number of years in current dhdt period (for color pallete)
    y0, y1  = [int(x) for x in dhdt_period.split('-')]
    dy=y1-y0
    
    
    ## add empty rows with delta_yr ranging from 1 to 20, to have same color range for all glaciers even if some glaciers don't have stakes with the same dy
    for delta_yr in range(1, dy+1):
        if delta_yr not in pd_stake_data_yyyy_avg['delta_yr'].values:
            empty_row = pd.DataFrame({
                'matched_stakeID': [np.nan],
                'mb_we_mean': [np.nan], 'mb_we_min': [np.nan], 'mb_we_max': [np.nan], 'mb_we_count': [0],
                'mb_we_std': [np.nan], 'mb_we_pct05': [np.nan], 'mb_we_pct95': [np.nan],
                f'mb_{best_approach}_buffer_mean': [np.nan],
                f'mb_{best_approach}_point_mean': [np.nan],
                f'mb_{best_approach}_buffer_std': [np.nan],
                'date0_dt': [pd.to_datetime('2000-01-01')],  # dummy date
                'date1_dt': [pd.to_datetime('2000-01-01')],  # dummy date
                'date0_yr': [2000],  # dummy year
                'delta_yr': [delta_yr]  # add missing delta_yr value
            })
            pd_stake_data_yyyy_avg = pd.concat([pd_stake_data_yyyy_avg, empty_row], ignore_index=True)
    
    ''' plot '''

    ## Rename  "mb_f001_point" to "mb_f001" to use in plotting function
    pd_stake_data_yyyy_avg = pd_stake_data_yyyy_avg.rename(
            columns={ f'mb_{best_approach}_point': f'mb_{best_approach}' })
    
    # fig,ax=plt.subplots(1, figsize=(5,6) )
    ax=ax3
    cbar_width = 0.01
    markersize = 200 # 100, 80
    # cmap_nr_obs = sns.color_palette("icefire" ,n_colors=24 )[:20]; norm = plt.Normalize(1,24)
    cmap_nr_obs = sns.color_palette("cubehelix_r" ,n_colors=40 )[10:30]; norm=plt.Normalize(1,20)

    # horiz_errorvar = 'mb_uncertainty'
    horiz_errorvar = 'mb_pct_range'

    _ , ax = evalF.plot_stake_smb_scatter_ax(pd_stake_data_yyyy_avg, best_approach, #glacier_sgiid, glacier_name, #dhdt_period, 
                                xvarname='mb_we_mean',errname=horiz_errorvar,
                                N_F_str = f'N={best_N}, F={best_F}', msize=markersize,
                                hue_var='delta_yr', style_var=None, # 'delta_yr',
                                plot_palette=cmap_nr_obs,
                                rmse_value=None, # rmse_values[best_approach], 
                                add_legend=False,
                                ax=ax)
    ## remove title
    ax.set_title(''); ax.set_xlabel('Glamos point MB'); #ax.set_ylabel('')

    cmap_add = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', cmap_nr_obs, N=20)
    # sm = plt.cm.ScalarMappable(cmap='icefire', norm=norm)
    sm = plt.cm.ScalarMappable(cmap=cmap_add, norm=norm)
    
    sm.set_array([])
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0 + 0.1, cbar_width, ax.get_position().height / 1.5])
    ax.figure.colorbar(sm, cax=cax, label='Number of observations', ticks=[1,5,10,15,20], orientation='vertical')

    ax.grid('major', linestyle='--', alpha=0.5)
    if glacier_sgiid == 'B36-26': ## aletsch: only one with multiple floating point precisiion on y-axis: fix.
        ## if ax y-label is using .1 float precision, update to .0 precision
        yticks = ax.get_yticks(); xticks = ax.get_xticks()
        ## set precision of yticks to no decimals
        yticks = xticks; xlim = ax.get_xlim(); 
        ax.set_yticks(yticks); ax.set_ylim(xlim)

    ## annote RMSE value in a corner with a box around it, in the bottom-right corner
    if plot_rmse_value:
        ax.text(0.90, 0.05, f'RMSE={rmse_stakes_bestparams:.2f}', transform=ax.transAxes,
                fontsize=fsize_ticklabels,
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle='round', edgecolor='black', facecolor='white',alpha=0.8))

        
    #### SAVE FIGURE 
    # fig.tight_layout()
    if save_fig:
        filename = f'{glacier_sgiid}_{glacier_rgiid}_{glacier_name}_{dhdt_period}_performance_bestParams-{best_approach}_N{best_N}_F{best_F_str}.pdf'
        fig.savefig(os.path.join(path2save_figure,filename ), dpi=300, bbox_inches='tight')

    del pd_stake_data_yyyy_avg
    # 
## concat dataframe with rmse values
df_all_glaciers = pd.concat(list_df_glacier_values, ignore_index=True)
df_all_glaciers
