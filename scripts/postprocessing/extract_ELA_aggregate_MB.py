#%%
## extract ELA from MB field; aggregate glacier-wide values to store as CSV
import ast
import os
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import rasterio as rio
import warnings
# import seaborn as sns
from tqdm import tqdm

import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.abspath('../scripts')))
import myFunctions as myf
import evalFunctions as evalF

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
path2mb ='../../data/bestParams'
path2montecarlo = os.path.join(data_dir,'SMB2D/bruteForce/bestParams_f001_F080_N9/uncertaintyMonteCarlo/')

path2save_figure = '../../figures/'
# data_dir = '../data'

my_palette = ['#2b6f39','#efbb1a','#d490c6'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#efbb1a


#%% Load data

'''------------------
## Load data
------------------'''
file_grab_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/IceElevation_50m_epsg32632.tif') 
da_grab_elev = xr.open_dataarray(file_grab_elevation).isel(band=0)
da_grab_elev_swiss = da_grab_elev.rio.reproject(swiss_crs, inplace=True)


'''------------------
## Glacier outlines (RGI shapefiles)
------------------'''

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs)



''' ################################################################
Load GLAMOS data
#################################################################### '''

## MB elevation-binned
path2data_glamos = os.path.join(data_dir,'GLAMOS/massbalance_fixdate_elevationbins.csv')
df_glamos_mb = myf.load_glamos_elevationbins(path2data_glamos)
path2data_glamos_fixdate = os.path.join(data_dir, 'GLAMOS/massbalance_fixdate.csv')
df_glamos_fixdate = evalF.load_glamos_fixdate(path2data_glamos_fixdate)

## SGI info 
glamos_traintest_file = os.path.join('../../files/glamos_train_test_split.csv')
df_glamos_sgi = pd.read_csv(glamos_traintest_file, index_col=0)
df_glamos_sgi['RGI_match'] = df_glamos_sgi['RGI_match'].apply(ast.literal_eval)
## update RGI column
df_glamos_sgi['RGI_unique'] = df_glamos_sgi['RGI_match'].apply(lambda x: x[0] if len(x)>0 else None)
df_glamos_sgi.loc[df_glamos_sgi.index=='A50i-19', 'RGI_unique'] = 'RGI60-11.008merged'
df_glamos_sgi.rename(columns={'name':'glacier_name'}, inplace=True)

df_glamos_sgi.reset_index(inplace=True) ## SGIID as column

rgi_list_glamos = df_glamos_sgi['RGI_unique'].dropna().unique().tolist() ## get list of RGI glaciers that have GLAMOS matches


#%% Process glaciers

dhdt_period = '2000-2020'

## get filelist for dhdt period
fpath_smb_bestparams = os.path.join(path2mb, f'mb_{dhdt_period}')
files_smb_list = os.listdir(fpath_smb_bestparams)
files_smb_list = [f for f in files_smb_list if f.endswith('.tif')]
files_smb_list.sort()

## DEMO: for Aletsch and Findel glaciers
files_aletsch_findel = ['RGI60-11.01450_mb-mwe_f001_F080_N9.tif',
                        'RGI60-11.02773_mb-mwe_f001_F080_N9.tif']; 
rgi_aletsch_findel = ['RGI60-11.01450', 'RGI60-11.02773']

## uncertainty files from MonteCarlo simulations (if available)
file_mc_fdiv = os.path.join(path2montecarlo,'MC_velocity-only/', f'swiss_{dhdt_period}_smb_std_MC-100_veloOnly-L250m.tif')
file_mc_dhdt = os.path.join(path2montecarlo,'MC_dhdt-only/', f'swiss_{dhdt_period}_smb_std_MC-100_dhdtOnly-L1000m.tif')
file_mc_mb = os.path.join(path2montecarlo,'MC_combined/', f'swiss_{dhdt_period}_smb_std_MC-100_combinedPerturbation_velo-250m_dhdt-1000m.tif')
da_mb_unct_swiss_fdiv = None; da_mb_unct_swiss_dhdt = None; da_mb_unct_swiss_combined = None; 
if os.path.exists(file_mc_fdiv):
    da_mb_unct_swiss_fdiv = xr.open_dataarray(file_mc_fdiv).isel(band=0).drop_vars('band')
if os.path.exists(file_mc_dhdt):
    da_mb_unct_swiss_dhdt = xr.open_dataarray(file_mc_dhdt).isel(band=0).drop_vars('band')
if os.path.exists(file_mc_mb):
    da_mb_unct_swiss_combined = xr.open_dataarray(file_mc_mb).isel(band=0).drop_vars('band')


'''## toggle All glaciers or demo set:'''
process_all_glaciers = True
files_to_process = files_smb_list if process_all_glaciers else files_aletsch_findel

plot_approach_figs = False
list_glacier_values=[]

pd_glacier_px_mb = pd.DataFrame([])

print(f'\n ######################################## \n ' \
          f'Processing glaciers')

# for fname in files_to_process:
for idx in tqdm(range(len(files_to_process)), desc="Processing glaciers"):
    fname = files_to_process[idx]
    
    ''' -------------------
    ## Glacier outlines
    -------------------'''

    glacier_rgiid = fname.split('_')[0]
    if glacier_rgiid not in gdf_swiss_large['RGIId'].values:
        print(f'Glacier {glacier_rgiid} is not in the selected RGI outlines; skipping')
        continue
    
    gdf_current_glacier = gdf_swiss_large.loc[gdf_swiss_large['RGIId']==glacier_rgiid].copy()
    glacier_name = gdf_current_glacier['Name'].item()
    
    assert glacier_rgiid == gdf_current_glacier['RGIId'].item(), f"Glacier RGIId mismatch: {glacier_rgiid} != {gdf_current_glacier['RGIId'].item()}; something wrong in df selection"

    if glacier_name is None:
        glacier_name=''

    ''' ----------
    ## Load my MB 
    -------------- '''
    ##%%

    ## m we 
    da_mb = xr.open_dataarray( os.path.join(path2mb, f'mb_{dhdt_period}/', fname) ).isel(band=0).drop_vars('band').rename('MB')

    ## rerpoject to Swiss CRS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
        gdf_current_glacier.to_crs(swiss_crs, inplace=True)
        
        da_mb = da_mb.rio.reproject(swiss_crs, inplace=True)
        da_mb = da_mb.assign_attrs({'units':'m.w.e./yr'})

        da_elev = da_grab_elev_swiss.rio.clip(gdf_current_glacier.geometry, gdf_current_glacier.crs, drop=True, all_touched=True)
        da_elev = myf.reproject_match_grid(da_mb, da_elev).rename('elevation')

        if da_mb_unct_swiss_fdiv is not None:
            da_mb_unct_fdiv = myf.reproject_match_grid(da_mb, da_mb_unct_swiss_fdiv) 
        else: da_mb_unct_fdiv = None
        
        if da_mb_unct_swiss_dhdt is not None:
            da_mb_unct_dhdt = myf.reproject_match_grid(da_mb, da_mb_unct_swiss_dhdt) 
        else: da_mb_unct_dhdt = None
        
        if da_mb_unct_swiss_combined is not None:
            da_mb_unct_mb = myf.reproject_match_grid(da_mb, da_mb_unct_swiss_combined) 
        else: da_mb_unct_mb = None
        
    
    ## bin values
    bin_step = 100 # default for comparing to GLAMOS which uses 100 m steps
    mb_bin_means_f001, hmin_binned, bin_std_f001, smb_bin_sem_f001 = myf.bin_da_to_elevbins(da_mb, da_elev, binstep=bin_step)
    
    
    ## Skip first bin(s) if they have negative gradient (up to 3 bins), skip them
    binidx_start = myf.get_binidx_to_start(mb_bin_means_f001, 
                                       max_skip_bins=3, 
                                       max_fraction=0.15)
    h_to_skip = hmin_binned[binidx_start] ## skip the first bin with values, using right-edge of bin by indexing+1
    
    ## store glacier MB pixel values in a dataframe to get regional statistics later
    pd_mb_glacier = da_mb.to_dataframe().reset_index().dropna(subset='MB')
    pd_glacier_px_mb = pd.concat([pd_glacier_px_mb, pd_mb_glacier['MB'] ], ignore_index=True)

    ''' -----------
    #########################
    ## Segment MB field into accumulation and ablation areas, using watershed segmentation
    Then calculate ELA based on linear regression on the points in each segment

    Watershed segmentation:
    - creates grouped regions based on data value split (MB=0)
    - remove small groups (small_obj_threshold) to avoid noise; set threshold as a function of glacier size, so that smaller glaciers have a lower threshold
    - creates mask (accumulation=1, ablation=-1) for further processing
    #########################
    ------------- '''
    # small_obj_threshold = 40 # pixels; remove small objects from segmentation
    ## set threshold as a function of the glacier size, so that smaller glaciers have a lower threshold (since they have less pixels)
    small_obj_threshold = min(40, int(da_mb.size * 0.01)) # 1% of glacier size; max 40 pixels

    da_segmented_binary = myf.get_watershed_segmentation(
        da_mb,
        segment_threshold=0,
        small_obj_threshold=small_obj_threshold,
    )

    
    ## use the segmented binary mask to extract acc/abl points and do linear regression to find ELA
    ds_glacier = xr.merge([da_mb, da_elev, da_segmented_binary.rename('segment')],compat='no_conflicts')

    ##%% to dataframe
    df_glacier = ds_glacier.to_dataframe().reset_index().dropna(subset=['MB','elevation','segment'])

    df_glacier_skip = df_glacier.loc[df_glacier['elevation']<=h_to_skip].copy()
    df_glacier_clean = df_glacier.loc[df_glacier['elevation']>h_to_skip].copy()

    ''' -----------
    Extracting ELA:
        ELA0: initial estimate based on all a linear regression to all points with MB<0 (from the segmented mask). Taken as intersect at MB=0
        ELA1: refined estimate based on linear regression of all points below ELA0, and accumulation points above ELA0. Taken as intersect at MB=0

    Mass balance gradient:
        Based on ELA1, do a final datasplit into ablation and accumulation points.
        MB gradient is the slope of the linear fit based on all points below/above ELA1.
    ------------- '''

    ''' ### SPLIT STEP 0, get ELA 0  '''
    ## now split by SEGMENT instead of MB=0
    df_ablation0 = df_glacier_clean.loc[df_glacier_clean['segment']== -1 ].copy()
    df_accumulation0 = df_glacier_clean.loc[df_glacier_clean['segment']==1].copy()
    
    ## ELA0; set to NaN if out of bounds (i.e. below the lowest elevation of the glacier)
    MB_gradient_ablation0, ELA0, x_vals0, y_vals0 = myf.linregress_mb_elev(df_ablation0)
    if ELA0 < df_glacier['elevation'].min(): ## ELA0 is below the lowest elevation of the glacier. Try first with ALL points, otherwise set to NaN
        df_ablation0 = df_glacier.copy()
        MB_gradient_ablation0, ELA0, x_vals0, y_vals0 = myf.linregress_mb_elev(df_ablation0)
    if ELA0 < df_glacier['elevation'].min(): ## ELA0 is still below the lowest elevation of the glacier. Tried first with ALL points, now set to NaN
        ELA0 = np.nan
        MB_gradient_ablation0 = np.nan

    ''' -----------
    SPLIT STEP 1, get ELA 1
    - use ELA0 to make update df_ablation points 
    - start again from df_allpoints (select all points <ELA0, so can include mb>0 points if they are below ELA0)
    --------------- '''
    ### Refine ELA using all pixels below the current estimate.
    df_ablation1     = df_glacier_clean.loc[df_glacier_clean['elevation'] < ELA0].copy()
    df_accumulation1 = df_glacier_clean.loc[df_glacier_clean['elevation'] >= ELA0].copy()
    MB_gradient_ablation1, ELA1, x_vals1, y_vals1 = myf.linregress_mb_elev(df_ablation1)
    
    MB_gradient_ablation2, MB_gradient_accumulation2 = np.nan, np.nan ## initialize; will overwrite if next step successfull.
    if not df_ablation1.empty:
        ## Refit below the updated ELA for the final ablation gradient.
        df_ablation2     = df_glacier_clean.loc[df_glacier_clean['elevation'] < ELA1].copy()
        df_accumulation2 = df_glacier_clean.loc[df_glacier_clean['elevation'] >= ELA1].copy()
        MB_gradient_ablation2, _, _, _ = myf.linregress_mb_elev(df_ablation2)
        if not df_accumulation2.empty:
            MB_gradient_accumulation2, _, _, _ = myf.linregress_mb_elev(df_accumulation2)
            
    ## check: if ELA1 is below the lowest elevation of the glacier, then set to NaN
    if ELA1 < df_glacier['elevation'].min():
        ELA1 = np.nan
        MB_gradient_ablation2 = np.nan
        MB_gradient_accumulation2 = np.nan
    
    ## other check: if ELA1 is above the max elevation of the glacier (+ a buffer of 1000 m), then set to NaN
    if ELA1 > df_glacier['elevation'].max() + 1000:
        ELA1 = np.nan
        MB_gradient_ablation2 = np.nan
        MB_gradient_accumulation2 = np.nan

    ''' -----------
    Final MB mask split: 
    using ELA1 to split the glacier into ablation and accumulation areas, for plotting and AAR calculation 
    ----------------'''
    if not np.isnan(ELA1):
        ## update segmented mask with final split using ELA1
        da_mb_acc_abl_mask = da_mb.copy()
        da_mb_acc_abl_mask = xr.where(da_elev >= ELA1, 1 , -1)   # Set values above ELA1 to 1, below ELA2 to -1
        da_mb_acc_abl_mask = da_mb_acc_abl_mask.where(~np.isnan(da_mb), np.nan) ## set 0 values to NaN for better plotting
    else:
        da_mb_acc_abl_mask = da_segmented_binary.copy() ## if ELA1 is NaN, then use the original segmented mask


    ''' -----------
    Estimate other aggregated values
    - AAR: use the segmented mask to calculate the area of accumulation vs ablation; use the LAST mask (split by ELA1)
    - mean values of glacier MB
    - median values of monteCarlo Uncertainty of glacier pixels
    - add some general values from RGI information and GLAMOS information (if available)
    ------------- '''
    
    total_px_count = int(da_mb_acc_abl_mask.count().item())
    acc_px_count = int((da_mb_acc_abl_mask == 1).sum().item())
    AAR = acc_px_count / total_px_count * 100 if total_px_count else np.nan

    median_unct_fdiv = da_mb_unct_fdiv.median().item() if da_mb_unct_fdiv is not None else np.nan
    median_unct_dhdt = da_mb_unct_dhdt.median().item() if da_mb_unct_dhdt is not None else np.nan
    median_unct_mb = da_mb_unct_mb.median().item() if da_mb_unct_mb is not None else np.nan
    valid_px_count = da_mb.count().values

    glacier_values = {
        'glacier_rgiid': glacier_rgiid,
        'glacier_name': glacier_name,
        'mean_MB': np.round(da_mb.mean().item(), 2),
        'min_MB': np.round(da_mb.min().item(), 2),
        'max_MB': np.round(da_mb.max().item(), 2),
        'median_unct_fdiv': np.round(median_unct_fdiv, 2),
        'median_unct_dhdt': np.round(median_unct_dhdt, 2),
        'median_unct_mb': np.round(median_unct_mb, 2),
        'ELA': np.round(ELA1,2),
        'MBgrad_abl': np.round(MB_gradient_ablation2,2),
        'MBgrad_acc': np.round(MB_gradient_accumulation2,2),
        'AAR': np.round(AAR, 2),
        'processed_area_km2': np.round(valid_px_count * np.abs(da_mb.rio.resolution()[0] * da_mb.rio.resolution()[1]) * 1e-6, 1), ## km2
        ## additional glacier info
        'min_elev_Grab': np.round(da_elev.min().item(), 1),
        'max_elev_Grab': np.round(da_elev.max().item(), 1),
        'mean_elev_Grab': np.round(da_elev.mean().item(), 1),
        'RGI_Area': gdf_current_glacier['Area'].sum(),
        'RGI_Lmax': gdf_current_glacier['Lmax'].max(),
        'RGI_Slope': gdf_current_glacier['Slope'].mean(),
        'RGI_Zmin': gdf_current_glacier['Zmin'].min(),
        'RGI_Zmax': gdf_current_glacier['Zmax'].max(),
        'sgi_id': None,
    }

    ##%% 
    ''' -----------
    #########################
    ## If GLAMOS glacier: also get GLAMOS ELA
    (A) calculate ELA from GLAMOS binned data (fixdate_elevbins), in the same way as my own. 
    (B) extract ELA from GLAMOS reported ELA (fixdate) (see glamos_aggregate_AAR.py script)
    #########################
    ------------- '''
    
    if glacier_rgiid in rgi_list_glamos:
        # print('--- split binned & GLAMOS  ---- ')

        glamos_y0 = int(dhdt_period.split('-')[0]) # 2000

        ''' ## GLAMOS ELA '''
        glacier_sgiid = df_glamos_sgi.loc[df_glamos_sgi['RGI_unique']==glacier_rgiid, 'sgi-id'].item() ## get glacier SGIId from GLAMOS SGI shapefile
        df_glamos_yyyy = df_glamos_mb.loc[ df_glamos_mb['date_start'].dt.year > glamos_y0].copy()
        df_mb_glacier, glac_name_ID = myf.get_glacier_mb_bins_glamos(df_glamos_yyyy, 
                                                glacier_id=glacier_sgiid)
        hmin_bin_glamos = df_mb_glacier.index.to_list() # x-axis values: glamos is 100m bin step
        glacier_name = df_glamos_sgi.loc[df_glamos_sgi['RGI_unique']==glacier_rgiid, 'glacier_name'].item() ## get glacier name from GLAMOS SGI shapefile
        glacier_values['sgi_id'] = glacier_sgiid
        glacier_values['glacier_name'] = glacier_name

        ## elevation-binned glamos values
        df_glamos_glacier = df_glamos_mb.loc[ (df_glamos_mb['sgi-id'] == glacier_sgiid)
                                                & (df_glamos_mb['date_start'].dt.year >= glamos_y0)  
                                                & (df_glamos_mb['date_start'].dt.year <= 2020)]
        
        ## fixdate (single glacier value) glamos values
        df_glamos_glacier_fixdate = df_glamos_fixdate.loc[(df_glamos_fixdate['sgi-id'] == glacier_sgiid)
                                        & (df_glamos_fixdate['date_start'].dt.year >= glamos_y0)
                                        & (df_glamos_fixdate['date_start'].dt.year <= 2020)]

        h_min_values = df_mb_glacier.index  # x-axis values
        smb_glamos_binned =  df_mb_glacier[('Ba', 'mean')]/1000

        ## estimate ELA from elevation-binned values
        ela_glamos, hbin_zero_crossings_glamos = myf.calculate_ELA_binned(smb_glamos_binned.values, h_min_values.values)
        MB_gradient_abl_glamos, MB_gradient_acc_glamos, ELA_glamos_oob, _, _, _,_ = myf.calculate_MB_gradient_glamos(smb_glamos_binned.values, h_min_values.values, ela_glamos, skip_first_bin=False)

        values_dict_glamos = {
            'ELA_G_binned': np.round(ela_glamos,1),
            'ELA_G_fixdate': np.round(df_glamos_fixdate['ELA'].mean(),1),
            'MBgrad_G_abl': np.round(MB_gradient_abl_glamos*1000,2),
            'MBgrad_G_acc': np.round(MB_gradient_acc_glamos*1000,2) if MB_gradient_acc_glamos else np.nan,
            'mean_MB_glamos_fixdate': np.round(df_glamos_glacier_fixdate['Ba'].mean()/1000, 2),
            'mean_AAR_glamos_fixdate': np.round(df_glamos_glacier_fixdate['AAR'].mean(), 2),
        }
        
        if plot_approach_figs and glacier_rgiid in rgi_aletsch_findel: ## only plot for demo glaciers
            plt.rcParams.update({'font.size': 14})
            fig = plt.figure(figsize=(18, 9))
            grid = fig.add_gridspec(
                2, 2, width_ratios=[1.3, 2.35], wspace=0.18, hspace=0.10
            )
            ax_ela0 = fig.add_subplot(grid[0, 0])
            ax_ela1 = fig.add_subplot(grid[1, 0])
            ax_glamos = fig.add_subplot(grid[:, 1])
            # ax_mb = fig.add_subplot(grid[:, 2])

            '''## ELA0 split/regression'''
            ax_ela0 = evalF.plot_ela_split_regression_ax(ax_ela0, df_ablation0, df_accumulation0, df_glacier_skip, 
                                                   x_vals0, y_vals0, ELA0, linestyle_ela = '--')
            ax_ela0.set_xticks([]); ax_ela0.set_xlabel(''); ax_ela0.set_title('ELA0 split / regression')
            ax_inset0 = evalF.plot_add_inset_mb_segmented(ax_ela0, da_segmented_binary, title='Initial segmentation')
            
            '''## ELA1 split/regression'''
            
            ax_ela1 = evalF.plot_ela_split_regression_ax(ax_ela1, df_ablation1, df_accumulation1, df_glacier_skip, 
                                                   x_vals1, y_vals1, ELA1, linestyle_ela = '-')
            ax_ela1.vlines(ELA0, *ax_ela1.get_ylim(), color=my_palette[0],
                           linestyle='--', alpha=0.5, label='ELA0')
            ax_ela1.set_title('ELA1 split / regression')
            ax_ela1.set_xlim(ax_ela0.get_xlim())
            ax_inset1 = evalF.plot_add_inset_mb_segmented(ax_ela1, da_mb_acc_abl_mask, title='Final segmentation')


            '''## plot GLAMOS '''
            ax = ax_glamos
            ax.fill_between(h_min_values, df_mb_glacier[('Ba', 'min')]/1000, df_mb_glacier[('Ba', 'max')]/1000, color='gray', alpha=0.2, label='GLAMOS min/max range')
            ax.plot(h_min_values,smb_glamos_binned, color='black', linestyle='-', label='GLAMOS mean')

            # Plot SMB 
            # 1. plot dash line for full solution
            ax.plot(hmin_binned, mb_bin_means_f001, linewidth=3, 
                    color=my_palette[0], linestyle='--', 
                    label=f'MB (excluded for ELA calc)', 
                    )
            # # 2. plot solid line for extent used to calculate gradient
            ax.plot(hmin_binned[hmin_binned>h_to_skip], mb_bin_means_f001[hmin_binned>h_to_skip], linewidth=3, 
                    color=my_palette[0], linestyle='-', 
                    label=f'MB', 
                    )
            ax.set_xlabel('Minimum elevation of bin [m a.s.l.]')
            ax.set_ylabel('Mass balance [m w.e./yr]')

            ## line for 0 SMB
            ax.hlines(0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color=[0.1,0.1,0.1], linestyle='--', alpha=0.5)
            ax.set_title(f'{glacier_sgiid}, {glacier_rgiid}, {glacier_name}')
            ax.grid('major', linestyle='--', alpha=0.5)

            ## add line for linreg ELA
            ax.axvline(ELA1, color=my_palette[0], linestyle='--', label=f'ELA1: {ELA1:.1f} m')
            ## add line for GLAMOS ELA
            if not np.isnan(ela_glamos): ax.axvline(ela_glamos, color='black', linestyle=':', label=f'GLAMOS ELA: {ela_glamos:.1f} m')
            
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')# fontsize='small')

            ## MB raster with final ELA1 boundary, ## add as inset to the GLAMOS plot
            ax_inset2 = ax_glamos.inset_axes([0.60, 0.04, 0.46, 0.42])
            ax = ax_inset2
            h = da_mb.plot.imshow(ax=ax, cmap='RdBu', vmin=-5, vmax=5, 
                              add_colorbar=True, cbar_kwargs={'label':'[m w.e./yr]', 'orientation':'vertical', 'shrink':0.2, 'pad':0.02})
            da_mb_acc_abl_mask.plot.contour(ax=ax, levels=[0], colors='black', linewidths=2)
            ## move colorbar to right of the inset
            cbar = h.colorbar; cbar.ax.set_position([0.93, 0.18, 0.03, 0.2])  # [left, bottom, width, height]
            ax.set_title('MB field and ELA1 boundary')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_aspect('equal')

            fig.suptitle(f'{glacier_sgiid}, {glacier_rgiid}, {glacier_name}')
            # fig.tight_layout()
            # fig.savefig(os.path.join(path2save_figure,'plot_extract_ELA', f'{glacier_rgiid}_ELA.png'), dpi=300)

    else:
        values_dict_glamos = { 'ELA_G': np.nan,
                    'MBgrad_G_abl': np.nan,
                    'MBgrad_G_acc': np.nan,
                    'mean_MB_glamos_fixdate': np.nan,
                    'mean_AAR_glamos_fixdate': np.nan}
    ## append glamos values to glacier_values
    glacier_values.update(values_dict_glamos)

    list_glacier_values.append(glacier_values)


df_glacier_values = pd.DataFrame(list_glacier_values)
print(' \n \n ----------------------- \n DONE ')


#%% save output to csv

## re-apply rounding some columns to two decimals
cols_to_round = ['ELA', 'MBgrad_abl','MBgrad_acc']
df_glacier_values[cols_to_round] = df_glacier_values[cols_to_round].round(2)
df_glacier_values = df_glacier_values.rename(columns={'processed_area_km2':'processed_area'})
col_order = ['glacier_rgiid','sgi_id', 'glacier_name', 'mean_MB','median_unct_fdiv','median_unct_dhdt','median_unct_mb', 'AAR','ELA', 'MBgrad_abl',    'MBgrad_acc',   'min_elev_Grab', 'max_elev_Grab',                                 'processed_area','RGI_Area']
col_unit_note = ['','','',                           'm.w.e./yr', 'm.w.e./yr',       'm.w.e./yr' ,      'm.w.e./yr',     '%', 'm a.s.l.', 'm.w.e./1000m', 'm.w.e./1000m', 'm a.s.l. (from Grab2021)', 'm a.s.l.(from Grab2021)', 'km2' , 'km2 (from RGI)']

df_to_save = df_glacier_values[col_order].copy()
## add row with unit
df_to_save.loc[-1] = col_unit_note  # adding a row
df_to_save.index = df_to_save.index + 1  # shifting index
df_to_save = df_to_save.sort_index()  # sorting by index

if len(df_glacier_values) == 100: ## only save if all glaciers were processed
    output_file = os.path.join( path2mb, f'aggregated_glacier_values_{dhdt_period}.csv')
    ## save only relevant columns
    df_to_save.to_csv(output_file, index=False)
    print(f'Saved aggregated glacier results to {output_file}')

#%%
''' ### print values '''

print(f'Aggregated MB statistics for {len(df_glacier_values)} glaciers')

print(f'\n-- Glacier-averages for ALL {len(df_glacier_values)} glaciers --')
print(f'Mean glacier MB         : {df_glacier_values["mean_MB"].mean():.2f} m w.e./yr ' 
                f'(glacier means range from {df_glacier_values["mean_MB"].min():.2f} to '
                f'{df_glacier_values["mean_MB"].max():.2f})')
# print(f'Pixel MB min-max range  : {df_glacier_values["min_MB"].min():.2f} to {df_glacier_values["max_MB"].max():.2f} m w.e./yr')
print(f'5-95th pctile of mean MB: '
    f'{df_glacier_values["mean_MB"].quantile(0.05):.2f} to '
    f'{df_glacier_values["mean_MB"].quantile(0.95):.2f} m w.e./yr')
print(f'Mean Monte Carlo unct   : {df_glacier_values["median_unctMB"].mean():.2f}  m w.e./yr')
print(f'Mean AAR (min-max)      : {df_glacier_values["AAR"].mean():.2f}%  ( {df_glacier_values["AAR"].min():.2f}% - {df_glacier_values["AAR"].max():.2f}% )')


negative_count = (df_glacier_values['mean_MB'] < 0).sum()
positive_count = (df_glacier_values['mean_MB'] >= 0).sum()
print(f'neg/positive glacierMB  : {negative_count} / {positive_count}' )

## print GLAMOS values for glaciers that have GLAMOS data
glamos_values = df_glacier_values.loc[df_glacier_values['sgi_id'].notna()]
print(f'\n-- Glacier-averages for GLAMOS ({len(glamos_values)}) glaciers --')
print('MB: mean derived MB / GLAMOS fixdate MB: \n'
    f'     {glamos_values["mean_MB"].mean():.2f} / '
    f'{glamos_values["mean_MB_glamos_fixdate"].mean():.2f} m w.e./yr')
print('ELA: derived ELA / GLAMOS elevation-bin ELA / GLAMOS fixdate ELA: \n'
    f'     {glamos_values["ELA"].mean():.2f} / '
    f'{glamos_values["ELA_G_binned"].mean():.2f} / '
    f'{glamos_values["ELA_G_fixdate"].mean():.2f} m ')
print('AAR: derived AAR / GLAMOS fixdate AAR : \n'
        f'     {glamos_values["AAR"].mean():.2f} % / '
        f'{glamos_values["mean_AAR_glamos_fixdate"].mean():.2f}% / ')


## Regional Swiss-wide MB statistics
print(f'\n-- Regional pixel-wise averages --')

print(
    f'Swiss-wide MB mean :  {pd_glacier_px_mb.mean().item():.2f} m w.e./yr \n'
    f'min-max            : {pd_glacier_px_mb.min().item():.2f} to {pd_glacier_px_mb.max().item():.2f} m w.e./yr \n'
    f'1-99 percentile    :  {pd_glacier_px_mb.quantile(0.01).item():.2f} to  {pd_glacier_px_mb.quantile(0.99).item():.2f}; \n'
    f'5-95 percentile    :  {pd_glacier_px_mb.quantile(0.05).item():.2f} to  {pd_glacier_px_mb.quantile(0.95).item():.2f} m w.e./yr \n'
)

'''## Integrate to Gt /yr
- 1 m w.e. is equivalent to 1000 kg m⁻² (i.e. 1 m of water)
- Mass change (Gt yr-1 )= MB (m w.e./yr) x Area (km2) x 10-3
'''
pixel_size_m = 50
pixel_area_km2 = pixel_size_m ** 2 * 1e-6
integrated_mb_gt = pd_glacier_px_mb.sum().item() * pixel_area_km2 * 1e-3
print(f'Swiss-wide integrated MB: {integrated_mb_gt:.2f} Gt/yr')
