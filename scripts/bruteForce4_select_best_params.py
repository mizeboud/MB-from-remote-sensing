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
# January 2026
'''
#%%
import ast

import numpy as np
import xarray as xr
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import myFunctions as myf
import evalFunctions as evalF

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
# data_dir = '../data'

path2glacier_output = os.path.join(data_dir,'bruteForceTMP/glaciers/')
path2saveRSME = os.path.join(data_dir, 'bruteForceTMP/rmse2glamos/')
path2save_bestParams = os.path.join(data_dir,'bruteForceTMP/bestParams/')


#%% load GLAMOS SGI info
''' ##################################################
Load GLAMOS SGI glacier info
###################################################### '''

glamos_traintest_file = os.path.join('../files/glamos_train_test_split.csv')
glamos_traintest = pd.read_csv(glamos_traintest_file, index_col=0)
glamos_traintest['RGI_match'] = glamos_traintest['RGI_match'].apply(ast.literal_eval)
glamos_traintest.rename(columns={'name':'glacier_name'}, inplace=True)
# df_glamos_sgi.reset_index(inplace=True) ## SGIID as column


'''------------------
## Glacier outlines (RGI shapefiles)
------------------'''

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs)



#%% Load calculated RMSE for all (N,F,glacier) for every GLAMOS glacier

''' ##################################################
Load calculated RMSE data
###################################################### '''
dhdt_period = '2000-2020'  # '2010-2020'  # '2000-2010'
# dhdt_period = '2010-2020'
# dhdt_period = '2015-2020'


path2rmse_output = os.path.join(path2saveRSME, f'dhdt_{dhdt_period}/')

## initialize list
list_rmse_elevbins_train, list_rmse_elevbins_test = [], []
list_rmse_stakes_train, list_rmse_stakes_test = [], []
list_rmse_total_train, list_rmse_total_test = [], []
## loop all GLAMOS glaciers
# for glacier_sgiid in glamos_traintest.index[-3:-2]:
for glacier_sgiid in glamos_traintest.index[[-3,-3]]:
    glacier_name = glamos_traintest.loc[glacier_sgiid,'glacier_name']
    traintest_subset = glamos_traintest.loc[glacier_sgiid,'set']
    print(f'.. loading {glacier_sgiid} ({glacier_name}), subset: {traintest_subset}')
    
    filelist_glacier_stakes = glob.glob( os.path.join(path2rmse_output, f'{glacier_sgiid}_*_weighted-rmse_*.nc') )
    filelist_glacier_elevbins = glob.glob( os.path.join(path2rmse_output, f'{glacier_sgiid}_*elevbins*.nc') )
    assert len(filelist_glacier_stakes)==len(filelist_glacier_elevbins) == 1
    ds_rmse_elevbins = xr.open_dataset( filelist_glacier_elevbins[0] )
    ds_rmse_stakes = xr.open_dataset( filelist_glacier_stakes[0] )
    
    '''# data (n_glacier, N, F); combine N=0 to the N=1-10 crowd by summing with f000'''

    ds_rmse_elevbins['rmse_f001'] = ds_rmse_elevbins['rmse_f001'].fillna(ds_rmse_elevbins['rmse_f000'])
    ds_rmse_elevbins['rmse_f110'] = ds_rmse_elevbins['rmse_f110'].fillna(ds_rmse_elevbins['rmse_f000'])

    ds_rmse_stakes['rmse_f001'] = ds_rmse_stakes['rmse_f001'].fillna(ds_rmse_stakes['rmse_f000'])
    ds_rmse_stakes['rmse_f110'] = ds_rmse_stakes['rmse_f110'].fillna(ds_rmse_stakes['rmse_f000'])

    ## store in list so I can stack (N,F) to (glacier,N,F) later
    if traintest_subset=='train':
        list_rmse_elevbins_train.append( ds_rmse_elevbins )
        list_rmse_stakes_train.append( ds_rmse_stakes )
    elif traintest_subset=='test':
        list_rmse_elevbins_test.append( ds_rmse_elevbins )
        list_rmse_stakes_test.append( ds_rmse_stakes )

## combine all glaciers along new dimension 'glacier'
ds_rmse_elevbins_train = xr.concat( list_rmse_elevbins_train, dim='glacier' ) # (glacier, N, F)
ds_rmse_stakes_train = xr.concat( list_rmse_stakes_train, dim='glacier' )

ds_rmse_elevbins_test = xr.concat( list_rmse_elevbins_test, dim='glacier' )
ds_rmse_stakes_test = xr.concat( list_rmse_stakes_test, dim='glacier' )


'''## calculate RMSE_MEAN and RMSE_TOTAL '''
ds_rmse_mean_train = (ds_rmse_elevbins_train + ds_rmse_stakes_train) / 2
ds_rmse_mean_test = (ds_rmse_elevbins_test + ds_rmse_stakes_test) / 2

ds_rmse_total_train = ds_rmse_elevbins_train + ds_rmse_stakes_train
ds_rmse_total_test = ds_rmse_elevbins_test + ds_rmse_stakes_test

## get N and F values
F_values = ds_rmse_total_train.Fparam.values
N_values = ds_rmse_total_train.Nlength.values

## update values to have 2 decimal points
F_values_flt = F_values.round(2)
F_values = [f"{fval:.2f}" for fval in F_values]


#%%

## annotate all N-F values in very small font, an :.1f
def annotate_heatmap(ax, arr, ndecimals=1,fsize=10, color='black'):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{arr[i, j]:.{ndecimals}f}", color=color, ha='center', va='center', fontsize=fsize)



''' ##################
Plot heatmaps (only for "train" Glaciers)
- use these to decide on the best N-F parameters for each fXXX (f001, f110, f111)
###################### '''

savefig = False ## toggle for development/final save


## Get f111 also for plot
ds_f111_plot = ds_rmse_mean_train['rmse_f111']#.sel(Fparam=1., method='nearest') # .mean(dim='glacier').round(2) #.values # Shape: (Ng, Nf, F)
## For F111: is 4D so for plotting 2D heatmaps select single F and plot the heatmap as same for F001 and F110
Ng_values = np.arange(1,6,1)
Nf_values = np.arange(1,6,1)
ds_f111_plot = ds_f111_plot.assign_coords(Ngrad=Ng_values, Nfdiv=Nf_values)
Ng_sel_value = 1;
Nf_sel_value = 5;
F_sel_value = 1.0 # 0.9


for ds_rmse_plot, rmse_plottype in [ 
    (ds_rmse_mean_train.copy(), 'RMSE-mean'), ## SELECT WHICH RMSE TO PLOT
    # (ds_rmse_elevbins_train.copy(), 'RMSE-elevbins'),
    # (ds_rmse_stakes_train.copy(), 'RMSE-stakes'),
    ]:
        
    # get minmax range
    rmse_mins = [ds_rmse_plot['rmse_f001'].median(dim='glacier').min(), ds_rmse_plot['rmse_f110'].median(dim='glacier').min()]
    rmse_maxs = [ds_rmse_plot['rmse_f001'].median(dim='glacier').max(), ds_rmse_plot['rmse_f110'].median(dim='glacier').max()]
    clim = [min(rmse_mins), max(rmse_maxs)]
    clim = [0, 3] ## from RMSE_total_train; set also for plots with only RMSE-elevbins or RMSE-stakes

    ''' ################################
    Plot heatmap for Mean(RMSE_bins & RMSE_point) across glaciers
    - gives good indication of differing performance for N-F parameters
    - round to 1 decimal, since a very detailed RMSE is not robust; and it gives a heatmap where we can select multiple best N-F
    ################################ '''
    aggtype = 'mean'; #clim = [0.7, 1.7]
    clim=[0,3]
    heatmap_dataf001_train = ds_rmse_plot['rmse_f001'].mean(dim='glacier').round(2) .values # Shape: (N, F)
    heatmap_dataf110_train = ds_rmse_plot['rmse_f110'].mean(dim='glacier').round(2) .values  # Shape: (N, F)

    fig, axs = plt.subplots(3,2, figsize=(15,18)) ## v1
    annotate_fz = 12; annotate_color = 'white';
    ## set global figure fontsize
    plt.rcParams.update({'font.size': 14})

    ''' -----------------
    plot f001 and f110 heatmaps
    --------------------- '''
    axs_fXXX = axs[:2,0] # axs[0,:2]
    _,_, min_value_f001, min_value_f110 = evalF.plot_heatmap(heatmap_dataf001_train.round(1), heatmap_dataf110_train.round(1), 
                                xlabel=F_values, ylabel=N_values, 
                                annotate_all=False,
                                clim=clim, cmap = sns.cubehelix_palette(as_cmap=True), 
                                axs=axs_fXXX # axs[0,:2]
                                )
    annotate_heatmap(axs_fXXX[0], heatmap_dataf001_train, ndecimals=2, fsize=annotate_fz, color=annotate_color)
    annotate_heatmap(axs_fXXX[1], heatmap_dataf110_train, ndecimals=2, fsize=annotate_fz, color=annotate_color)
    # fig.suptitle(f'{aggtype} {rmse_plottype} across trainset ({len(ds_rmse_plot.glacier)} glaciers)')
    # fig.suptitle(f'{rmse_plottype} across trainset ({len(ds_rmse_plot.glacier)} glaciers)')
    axs_fXXX[0].set_title(f'{aggtype} RMSE; smooth only fluxdiv (f001)')
    axs_fXXX[1].set_title(f'{aggtype} RMSE; smooth only gradients (f110)')


    ''' -----------------
    plot f111 heatmaps
    --------------------- '''
    ### F111 plot
    heatmap_Nf = ds_f111_plot.sel(Nfdiv=Nf_sel_value, method='nearest').mean(dim='glacier').round(1) .values # Shape: (N, F)
    heatmap_Ng = ds_f111_plot.sel(Ngrad=Ng_sel_value, method='nearest').mean(dim='glacier').round(1) .values # Shape: (N, F)
    heatmap_F = ds_f111_plot.sel(Fparam=F_sel_value, method='nearest').mean(dim='glacier').round(1) .values # Shape: (N, F)
    
    
    ax=axs[2,0] # axs[0,2]
    evalF.plot_heatmap_ax(heatmap_F, ax=ax,
                                xlabel=Ng_values, ylabel=Nf_values, 
                                annotate_all=False,
                                clim=clim, cmap = sns.cubehelix_palette(as_cmap=True), 
                                )
    annotate_heatmap(ax, heatmap_F, ndecimals=2, fsize=annotate_fz, color=annotate_color)
    # fig.suptitle(f'{rmse_plottype} across trainset ({len(ds_rmse_plot.glacier)} glaciers)')
    ax.set_title(f'{aggtype} RMSE; smooth both (f111), F={F_sel_value} ')
    ax.set_ylabel('N dimension (Ng)');  
    ax.set_xlabel('N dimension (Nf)')

    # ## get colormap from ax
    ## update ax size to be a bit more squashed vertically so that it matches the other plots (and I create more vertical space on the page)
    ax.set_aspect(0.4)


    ''' ################################
    Plot heatmap for RANGE (RMSE_bins & RMSE_point) across glaciers (min-max)
    - gives good indication of how robust RMSE value is between individual glaciers 
    ################################ '''

    aggtype = 'range'; #clim=[0.3, 2]
    clim=[0,5] 
    heatmap_dataf001_train = (ds_rmse_plot['rmse_f001'].max(dim='glacier') - ds_rmse_plot['rmse_f001'].min(dim='glacier')
                              ).round(2).values # Shape: (N, F)
    heatmap_dataf110_train = (ds_rmse_plot['rmse_f110'].max(dim='glacier') - ds_rmse_plot['rmse_f110'].min(dim='glacier')
                              ).round(2).values  # Shape: (N, F)  
    
    ''' -----------------
    plot f001 and f110 heatmaps
    --------------------- '''
    axs_fXXX = axs[:2,1] # axs[1,:2] axs[1,:2],
    _,_, min_value_f001, min_value_f110 = evalF.plot_heatmap(heatmap_dataf001_train.round(1), heatmap_dataf110_train.round(1), 
                                xlabel=F_values, ylabel=N_values, 
                                annotate_all=False,
                                clim=clim, cmap=sns.color_palette("crest", as_cmap=True),
                                axs=axs[:2,1] # axs[1,:2] axs[1,:2],
                                )
    annotate_heatmap(axs_fXXX[0], heatmap_dataf001_train, ndecimals=2, fsize=annotate_fz, color=annotate_color)
    annotate_heatmap(axs_fXXX[1], heatmap_dataf110_train, ndecimals=2, fsize=annotate_fz, color=annotate_color)
    axs_fXXX[0].set_title(f'{aggtype} RMSE; smooth only fluxdiv (f001)')
    axs_fXXX[1].set_title(f'{aggtype} RMSE; smooth only gradients (f110)')

    
    ''' -----------------
    plot f111 heatmaps
    --------------------- '''
    heatmap_F = (  ds_f111_plot.sel(Fparam=F_sel_value, method='nearest').max(dim='glacier') 
                  - ds_f111_plot.sel(Fparam=F_sel_value, method='nearest').min(dim='glacier') 
                ).round(1).values # Shape: (N, F)

    ax=axs[2,1]# axs[1,2]
    _, ax, min_value_ax = evalF.plot_heatmap_ax(heatmap_F, ax=ax,
                                xlabel=Ng_values, ylabel=Nf_values, 
                                annotate_all=False,
                                clim=clim, cmap = sns.color_palette("crest", as_cmap=True),
                                )
    annotate_heatmap(ax, heatmap_F, ndecimals=2, fsize=annotate_fz, color=annotate_color)
    # fig.suptitle(f'{rmse_plottype} across trainset ({len(ds_rmse_plot.glacier)} glaciers)')
    ax.set_title(f'{aggtype} RMSE; smooth both (f111), F={F_sel_value} ')
    ax.set_ylabel('N dimension (Nf)')
    ax.set_xlabel('N dimension (Ng)')
    
    # ## get colormap from ax
    ## update ax size to be a bit more squashed vertically so that it matches the other plots (and I create more vertical space on the page)
    ax.set_aspect(0.4)

    ## ## remove ALL colorbars from ax (all are 0-3 and I'll add it in Illustrator again)
    for ax in axs.flatten():
        cbar = ax.collections[0].colorbar
        cbar.remove() 


    ''' -----------------
    saving
    --------------------- '''
    fig.tight_layout()
    if savefig:
        path2save = f'../figures/evaluate_bruteForce/dhdt_{dhdt_period}/'
        figname = f"train_glaciers_{dhdt_period}_weighted{rmse_plottype}_mean-and-range_heatmap_fXXX"
        fig.savefig(os.path.join(path2save, figname+ '.pdf'))


#%% ## select single N/F and store RMSE for all glaciers

''' ##################################################
Summarize RMSE values per glacier for selected bestParameters

Best parameters: 
'f001': N= 9,   F= 0.80
'f110': N= 7,   F= 1
'f111': Ng= 1, Nf=5, F= 1

###################################################### '''

saveXLSX = False
# import json
best_param_dict = pd.read_json('../files/best_parameters_bruteforce_weighted.json')


table_fXXX = []
for fXXX in ['f001', 'f110','f111']:
    best_F = best_param_dict[fXXX]['F']
    
    df_rmse_NF = pd.DataFrame()
    for rmse_set, ds_rmse_train, ds_rmse_test in [
                # ('RMSE_total', ds_rmse_total_train, ds_rmse_total_test),
                ('RMSE_mean', ds_rmse_mean_train, ds_rmse_mean_test),
                ('RMSE_elevbins', ds_rmse_elevbins_train, ds_rmse_elevbins_test),
                ('RMSE_stakes', ds_rmse_stakes_train, ds_rmse_stakes_test),
                    ]:
        ## select best N/F gridcell of all glaciers
        if fXXX != 'f111':
                
            df_rmse_NF_set_train = (ds_rmse_train[[f'rmse_{fXXX}']] 
                                    .sel(Nlength=best_param_dict[fXXX]['N'],
                                        Fparam =best_param_dict[fXXX]['F'], method='nearest')
                                    .to_dataframe().reset_index()
                                    )
                
            df_rmse_NF_set_test = (ds_rmse_test[[f'rmse_{fXXX}']] 
                                    .sel(Nlength=best_param_dict[fXXX]['N'],
                                        Fparam =best_param_dict[fXXX]['F'], method='nearest')
                                    .to_dataframe().reset_index()
                                    )
        elif fXXX == 'f111':
            ## add 'Ng' and Nf as coordinates for f111
            ds_rmse_train_f111 = ds_rmse_train[['rmse_f111']].assign_coords( 
                        Ngrad = np.arange(1,6,1), Nfdiv =  np.arange(1,6,1))
            ds_rmse_test_f111 = ds_rmse_test[['rmse_f111']].assign_coords( 
                        Ngrad = np.arange(1,6,1), Nfdiv =  np.arange(1,6,1))

            df_rmse_NF_set_train = (ds_rmse_train_f111
                                    .sel(Ngrad =best_param_dict[fXXX]['Ng'],
                                        Nfdiv =best_param_dict[fXXX]['Nf'],
                                        Fparam=best_param_dict[fXXX]['F'], method='nearest')
                                    .to_dataframe().reset_index()
            )
            df_rmse_NF_set_test = (ds_rmse_test_f111
                                    .sel(Ngrad =best_param_dict[fXXX]['Ng'],
                                        Nfdiv =best_param_dict[fXXX]['Nf'],
                                        Fparam=best_param_dict[fXXX]['F'], method='nearest')
                                    .to_dataframe().reset_index()
            )

        df_rmse_NF_set = pd.concat( [df_rmse_NF_set_train, df_rmse_NF_set_test], ignore_index=True )
        
        ## rename rmse columns
        # df_rmse_NF_set = df_rmse_NF_set.rename(columns={'rmse_f001':f'{rmse_set}_f001', 'rmse_f110':f'{rmse_set}_f110'})
        df_rmse_NF_set = df_rmse_NF_set.rename(columns={f'rmse_{fXXX}':f'{rmse_set}_{fXXX}'}) # , 'rmse_f110':f'{rmse_set}_f110'})

        ## merge to main df
        if df_rmse_NF.empty:
            df_rmse_NF = df_rmse_NF_set
        else:
            df_rmse_NF[f'{rmse_set}_{fXXX}'] = df_rmse_NF_set[[f'{rmse_set}_{fXXX}']]

    ## add column as first columns:
    df_rmse_NF = df_rmse_NF.drop(columns=['glacier'])
    df_rmse_NF['glacier_sgiid'] = glamos_traintest.index
    df_rmse_NF['set'] = glamos_traintest['set'].values
    df_rmse_NF['glacier_name'] = glamos_traintest['glacier_name'].values

    ## reorder to have these added columsn as first
    if fXXX != 'f111': cols_to_order = ['glacier_sgiid', 'glacier_name', 'set','Nlength','Fparam']
    if fXXX == 'f111': cols_to_order = ['glacier_sgiid', 'glacier_name', 'set','Ngrad','Nfdiv','Fparam']
    new_columns = cols_to_order + (df_rmse_NF.columns.drop(cols_to_order).tolist())
    df_rmse_NF = df_rmse_NF[new_columns]

    table_fXXX.append( (fXXX, df_rmse_NF) )


if saveXLSX:
    fparam_str = f'{best_F:.2f}'.replace('.','')
    filename = f"RMSEweighted_train-test_glaciers_{dhdt_period}_bestparams.xlsx"
    ## write each fXXX to a separate sheet in the same excel file
    with pd.ExcelWriter(os.path.join(path2saveRSME, filename)) as writer:
        for fXXX, df in table_fXXX:
            df.to_excel(writer, sheet_name=f'{fXXX}', index=False)

# %%

''' ################################################################
Load glacier output for all parameter grid
Extract bestParams and save to tiff
#################################################################### '''


''' hugonnet dhdt '''
dhdt_hugo0020 = xr.open_dataarray(os.path.join(data_dir,
                'GlacierElevationChange_Hugonnet2021/11_rgi60_2000-01-01_2020-01-01/',
                'dhdt/dHdt_swiss_50m_epsg32632.tif')
                ).isel(band=0).rename('dhdt')

dhdt_hugo1020 = xr.open_dataarray(os.path.join(data_dir,
                'GlacierElevationChange_Hugonnet2021/11_rgi60_2010-01-01_2020-01-01/',
                'dhdt/dHdt_swiss_50m_epsg32632.tif')
                ).isel(band=0).rename('dhdt')

dhdt_hugo1520 = xr.open_dataarray(os.path.join(data_dir,
                'GlacierElevationChange_Hugonnet2021/11_rgi60_2015-01-01_2020-01-01/',
                'dhdt/dHdt_swiss_50m_epsg32632.tif')
                ).isel(band=0).rename('dhdt')


'''## get list of processed glaciers'''
rgi_dir_list = [dir for dir in os.listdir(path2glacier_output) if dir.startswith("RGI")] ## 99 glaciers -- 100 with RGImerged for Claridenfirn
rgi_dir_list.sort()
print(len(rgi_dir_list))
# glacier_rgiid = 'RGI60-11.00872'


best_approach = 'f001' # update this to select which approach to plot as thick line in the end
best_N = best_param_dict[best_approach]['N']; 
best_F = best_param_dict[best_approach]['F']
best_F_str = f'{best_F:.2f}'.replace('.','') ## for filename only: 080

save_tiff = False 
# save_tiff = True

## set up directories
os.makedirs(path2save_bestParams, exist_ok=True)
os.makedirs(os.path.join(path2save_bestParams, 'fluxdiv'), exist_ok=True)

for dhdt_period in ['2000-2020', '2010-2020', '2015-2020']:
    os.makedirs(os.path.join(path2save_bestParams, f'mb_{dhdt_period}'), exist_ok=True)

    for glacier_rgiid in rgi_dir_list:
        # print(f'-----------\n {glacier_rgiid}')
        # print(f".. processing {glacier_rgiid}")
        path2glacier = os.path.join(path2glacier_output, glacier_rgiid)

        gdf_glacier_rgi = gdf_swiss_large.loc[gdf_swiss_large['RGIId'] == glacier_rgiid].copy()

        ## Extract glacier elevation data and the RGI outlines 
        da_dhdt1520 = dhdt_hugo1520.rio.clip(gdf_glacier_rgi.geometry, drop=True,all_touched=True)
        da_dhdt1020 = dhdt_hugo1020.rio.clip(gdf_glacier_rgi.geometry, drop=True,all_touched=True)
        da_dhdt0020 = dhdt_hugo0020.rio.clip(gdf_glacier_rgi.geometry, drop=True,all_touched=True)
        dict_dhdt = {'2015-2020': da_dhdt1520, '2010-2020': da_dhdt1020, '2000-2020': da_dhdt0020}

        ## fluxdiv without density conversion (m i.e. /yr)
        ds_glacier_fdiv = xr.open_dataset(
            os.path.join(path2glacier,  f'{glacier_rgiid}_fluxdiv_paramgrid.nc' ))
        ds_glacier_fdiv.rio.write_crs('EPSG:32632', inplace=True)

        ## MB with density conversion ( m w.e. /yr)
        ds_glacier_mb = xr.open_dataset(
            os.path.join(path2glacier, 'mb', f'{glacier_rgiid}_mb_{dhdt_period}_paramgrid.nc' ))
        ds_glacier_mb.rio.write_crs('EPSG:32632', inplace=True)
        
        da_mb_bestparams   = ds_glacier_mb[f'mb_{best_approach}'].sel(Nlength=best_N, Fparam=best_F) ## epsg 32632
        da_fdiv_bestparams = ds_glacier_fdiv[f'fluxdiv_{best_approach}'].sel(Nlength=best_N, Fparam=best_F) ## epsg 32632
        da_fdiv_bestparams.attrs = ds_glacier_fdiv.attrs
        da_fdiv_bestparams.attrs['long_name'] = 'flux divergence'

        ## save geotiffs,
        if save_tiff:
            fname_fdiv = f'{glacier_rgiid}_fluxdiv-mie_{best_approach}_F{best_F_str}_N{best_N}.tif'
            da_fdiv_bestparams.rio.to_raster(os.path.join(path2save_bestParams, 'fluxdiv', fname_fdiv), driver='COG')

            fname_mb = f'{glacier_rgiid}_mb-mwe_{best_approach}_F{best_F_str}_N{best_N}.tif'
            da_mb_bestparams.rio.to_raster(os.path.join(path2save_bestParams, 
                                                        f'mb_{dhdt_period}', fname_mb), driver='COG')
        
