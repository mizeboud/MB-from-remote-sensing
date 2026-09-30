## Read GLAMOS mass-balance data and determine train/test split of glacierdata.

# author: M Izeboud
# August 2025
#%%
import xarray as xr
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Polygon

import sys
sys.path.append(os.path.dirname(os.path.abspath('../scripts')))
import myFunctions as myf
import evalFunctions as evalF

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'



# %% Load MB dataset to Eval to MB stakes
''' ################################################################
Load RGI shapefiles and information
#################################################################### '''

# '''## RGI inventory to mask glacier areas'''

# path2data = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/')
# gl_outline_file = '11_rgi60_CentralEurope.shp'

# gl_outline_shp_rgi11 = gpd.read_file(os.path.join(path2data,gl_outline_file))
# gl_outline_shp_rgi11 = gl_outline_shp_rgi11.to_crs(target_crs)


# ## swiss-region dataset estimataed from Grab 
# path2data = os.path.join(data_dir,'SwissGlacierThickness-R2020/') 
# swiss_regions_file = 'regions_approx_4326.shp'
# swiss_regions = gpd.read_file(os.path.join(path2data,swiss_regions_file))
# swiss_regions = swiss_regions.to_crs(target_crs) # ds_vx.rio.crs)

# # Reproject and clip shapefile to swiss bounds
# swiss_bounds = swiss_regions.total_bounds
# swiss_bounds_poly = gpd.GeoDataFrame(index=[0], crs=target_crs, geometry=[Polygon.from_bounds(*swiss_bounds)])  
# gl_outline_swiss = gpd.sjoin(gl_outline_shp_rgi11, swiss_bounds_poly).drop('index_right',axis=1) # intersection using geopandas SJOIN(left, right) clips LEFT to the RIGHT geom


# ''' ALL SWISS glaciers; use RGI outliens; only larger than >2km '''
# gdf_swiss_large = gl_outline_swiss.loc[gl_outline_swiss['Area']>2] # 117 glaciers
# # print(len(gdf_swiss_large))

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs)
rgi_claridenfirn = ['RGI60-11.00817', 'RGI60-11.00819', 'RGI60-11.00843', 'RGI60-11.00878']

gdf_swiss_all = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=0, target_crs=target_crs)



# %% Load MB dataset to Eval to MB stakes

''' ################################################################
Load GLAMOS data
#################################################################### '''


''' -------------------
Read GLAMOS MB of elevation bins
-------------------'''

# ## read massbalance elevation bins - GLAMOS
# path2data = os.path.join(data_dir,'GLAMOS/massbalance_fixdate_elevationbins.csv')
# # path2data = os.path.join(homedir,'data/GLAMOS/mass_balance_test.csv')
# df_glamos_raw = pd.read_csv(path2data, header=[6,7,8], delimiter=';')
# ## Swiss Glacier Inventory ID: sgi-id -- Aletsch is B36-26
# '''## pre-processing of dataframe'''
# df_glamos_mb = df_glamos_raw.copy()
# df_glamos_mb.columns = df_glamos_mb.columns.droplevel([0, 2])  # Keep only the first level (header at line 6) and drop the others
# df_glamos_mb.rename(columns={'Unnamed: 0_level_1':'name', '(according to Swiss Glacier Inventory)':'sgi-id','Unnamed: 11_level_1':'observer'}, inplace=True)
# df_glamos_mb['name'] = df_glamos_mb['name'].astype(str)
# df_glamos_mb['sgi-id'] = df_glamos_mb['sgi-id'].astype(str)
# for date_col in ['date_start','date_end_winter','date_end']:
#     df_glamos_mb[date_col] = pd.to_datetime(df_glamos_mb[date_col], format='%d/%m/%Y', errors='coerce').astype('datetime64[ns]')
# ## select years that overlap with Hugonnet period (2000-2019)
# df_glamos_mb = df_glamos_mb.loc[df_glamos_mb['date_start'].dt.year.isin(range(2015,2020))]
## UPDATE: need full 2000-2020 range, not 2015-2020 --> and moved to later in script, see below.
# # columns
# df_varnames = pd.DataFrame(
#                         data=np.array([
#                             df_glamos_raw.columns.droplevel([1,2]), 
#                             df_glamos_mb.columns, 
#                             df_glamos_raw.columns.droplevel([0,1])
#                         ]).T, 
#                         columns=['name', 'variable', 'unit']
#                     )
# dict_varnames = df_varnames.set_index('variable')[['unit', 'name']].to_dict(orient='index');
# unit = dict_varnames['Bs']['unit']
# unit = 'mm w.e.'

# '''## ['E22-16'] Vadret Pers in GLAMOS-MB is not included in GLAMOS SGI-shapefile. '''
# ## Instead, use Vadret da Morteratsch ['E22-03']
# # update sgi for morteratsch; keep name? 
# df_glamos_mb.loc[df_glamos_mb['sgi-id']=='E22-16','name'] = 'Vadret Pers and Vadret da Morteratsch'
# df_glamos_mb.loc[df_glamos_mb['sgi-id']=='E22-16','sgi-id'] = 'E22-03'


'''## fixdate glaciers (includes AAR)'''
path2data_glamos = os.path.join(data_dir,'GLAMOS/massbalance_fixdate.csv')
df_glamos_raw = pd.read_csv(path2data_glamos, header=[4,5,6], delimiter=';')
df_glamos_mb = evalF.load_glamos_fixdate(path2data_glamos) 

# '''## pre-processing of dataframe'''
# df_glamos_mb = df_glamos_raw.copy()
# df_glamos_mb.columns = df_glamos_mb.columns.droplevel([0, 2])  # Keep only the first level (header at line 6) and drop the others
# df_glamos_mb.rename(columns={'Unnamed: 0_level_1':'name', '(according to Swiss Glacier Inventory)':'sgi-id','Unnamed: 11_level_1':'observer'}, inplace=True)
# df_glamos_mb['name'] = df_glamos_mb['name'].astype(str)
# df_glamos_mb['sgi-id'] = df_glamos_mb['sgi-id'].astype(str)
# for date_col in ['date_start','date_end_winter','date_end']:
#     df_glamos_mb[date_col] = pd.to_datetime(df_glamos_mb[date_col], format='%Y-%m-%d', errors='coerce').astype('datetime64[ns]')
# df_glamos_mb
# '''## ['E22-16'] Vadret Pers in GLAMOS-MB is not included in GLAMOS SGI-shapefile. '''
# ## Instead, use Vadret da Morteratsch ['E22-03']
# # update sgi for morteratsch; keep name? 
# df_glamos_mb.loc[df_glamos_mb['sgi-id']=='E22-16','name'] = 'Vadret Pers and Vadret da Morteratsch'
# df_glamos_mb.loc[df_glamos_mb['sgi-id']=='E22-16','sgi-id'] = 'E22-03'


## some insights into which glaciers I can evaluate-- 28 if i select hugonnet-range,  32 if I allow all-time.
sgi_ids_glamos = df_glamos_mb['sgi-id'].unique()
glacier_names_glamos = df_glamos_mb['name'].unique()


## select dhdt period 2000-2020
df_glamos_mb
df_glamos_mb = df_glamos_mb.loc[ (df_glamos_mb['date_start'].dt.year >= 2000) & \
                                 (df_glamos_mb['date_start'].dt.year <= 2020) ].copy() # only data that is in same dhdt period

## convert mb values from mm we to mwe
df_glamos_mb[['Bw','Bs','Ba']] = df_glamos_mb[['Bw','Bs','Ba']]/1000.0

''' ## Using MB elevationbins and then averaging per glacier is not going to give glacier-wide MB values ofc . 
UPDATE
- need to work with massbalance_fixdate.csv instead of massbalance_fixdate_elevationbins.csv
- need to work with 2000-2020 instead of 2015-2020 timeframe. 
'''



# %% check all names in GLAMOS if they exist in RGI_OUTLINE info
''' ##############
Match GLAMOS SGI glaciers to RGI outlines
- creates "RGI_match" column; try to be strict (aiming for 1 match per Glamos glacier) so filter multiple matches
##################
'''
glacier_names_rgi = gdf_swiss_large['Name'].unique() ## many dont haave a name, only RGI-ID
rgi_ids = gdf_swiss_large['RGIId'].unique()


## ALL SWISS GLACIERS IDENTIFIED BY GLAMOS: 1400 
gdf_sgi_all = gpd.read_file(os.path.join(data_dir, 'GLAMOS/inventory_sgi2016_r2020/SGI_2016_glaciers.shp'))
gdf_sgi_all = gdf_sgi_all.to_crs(target_crs).drop(['pk_glacier','gid','rl_0','rl_1','rl_2','rl_3','i_code'],axis='columns') # ds_vx.rio.crs)

## Select all SGI glaciers where GLAMOS has MB data (in the hugonnet-timeframe) (28 glaciers)
gdf_sgi_mb = gdf_sgi_all.loc[gdf_sgi_all['sgi-id'].isin(sgi_ids_glamos)].copy()
print(f'.. {len(gdf_sgi_mb)} GLAMOS glaciers available')

## select only glaciers with area  > 2 km 
gdf_sgi_mb = gdf_sgi_mb.loc[gdf_sgi_mb['area_km2']>2].copy()
print(f'.. {len(gdf_sgi_mb)} GLAMOS glaciers larger than 2 km ')

'''## loop GLAMOS glacier outlines and find which RGI idx corresponds'''
for idx, df_row in gdf_sgi_mb.iterrows():
    
    # spatial join of dataframes : if geometries overlap, I can match them.
    gdf_row = gdf_sgi_mb.loc[[idx]][ ['sgi-id','name','year_acq','area_km2','geometry'] ].copy()
    
    gdf_sjoin = gdf_swiss_all[
                ['RGIId','Slope','Surging','Name','Area','geometry']
                ].sjoin(gdf_row).reset_index()

    
    
    if len(gdf_sjoin) == 0: # no matches
        print(gdf_row['name'].item(), '.. did not find overlap, increasing buffer ...')
        ## expand glamos geometry to have larger potential overlap area 
        gdf_glamos_buff = gdf_row.copy()
        gdf_glamos_buff['geometry'] = gdf_row.buffer(500)
        gdf_sjoin = gdf_swiss_all[
                    ['RGIId','Slope','Surging','Name','Area','geometry']
                    ].sjoin(gdf_glamos_buff).reset_index()
        if len(gdf_sjoin)>0:
            raise ValueError('... matching RGI only after applying buffer -- to check if correct, plot the glacier outlines and check if it is the right one.')

    # If too many matches, refine geometry overlap
    if len(gdf_sjoin)> 1: 
        ## shrink glamos geometry to avoid 'intersection' at edges and across ice divides
        gdf_glamos_buff = gdf_row.copy()
        gdf_glamos_buff['geometry'] = gdf_row.buffer(-100)
        gdf_sjoin = gdf_swiss_all[
                    ['RGIId','Slope','Surging','Name','Area','geometry']
                    ].sjoin(gdf_glamos_buff).reset_index()
        # print(f".. now {len(gdf_sjoin)}")
        if len(gdf_sjoin) == 0:
            raise ValueError(f"Glacier {gdf_row['name'].item()} ({gdf_row['sgi-id'].item()}) has no RGI match in Swiss outline after shrinking buffer. Check geometry or increase buffer size.")

    ## if still too many matches, continue with second step of shrinking (but dont do for Claridenfirn)
    buffer_step_m = 100
    max_buffer_m = 2000
    buffer_distance_m = 0

    while len(gdf_sjoin) > 1 and buffer_distance_m < max_buffer_m and not gdf_row['sgi-id'].item() == 'A50i-19': # Claridenfirn is excempt from this
        buffer_distance_m += buffer_step_m

        gdf_glamos_buff = gdf_row.copy()
        gdf_glamos_buff['geometry'] = gdf_row.geometry.buffer(-buffer_distance_m)

        if gdf_glamos_buff.geometry.is_empty.all():
            raise ValueError(
                f"Shrinking removed the GLAMOS geometry for "
                f"{gdf_row['name'].item()} ({gdf_row['sgi-id'].item()})"
            )

        gdf_sjoin_new = gdf_swiss_all[
            ['RGIId', 'Slope', 'Surging', 'Name', 'Area', 'geometry']
        ].sjoin(gdf_glamos_buff).reset_index()

        if gdf_sjoin_new.empty:
            raise ValueError(
                f"No RGI match remains after shrinking the GLAMOS geometry for "
                f"{gdf_row['name'].item()} ({gdf_row['sgi-id'].item()}) "
                f"by {buffer_distance_m} m"
            )

        gdf_sjoin = gdf_sjoin_new
    
    # if gdf_row['sgi-id'].item() == 'B82-27': # d'Otemma zentral
    #     # drop row with RGI60-11.02787, which is Bas Glacier d'Arolla
    #     gdf_sjoin = gdf_sjoin.loc[gdf_sjoin['RGIId']!='RGI60-11.02787'].reset_index(drop=True)
    
    # list of corresponding RGIIds (can be multiple sometimes)                
    glacier_rgiids = gdf_sjoin['RGIId'].unique()
    glacier_rgiids.sort()
    glacier_rgiids= list(glacier_rgiids)
    
    ## put list of RGIs in df
    gdf_sgi_mb.at[idx,'RGIId_match']=glacier_rgiids


## check indices for no-RGI match: 671; 1185
empty_rows = gdf_sgi_mb[gdf_sgi_mb['RGIId_match'].apply(lambda x: len(x) == 0)]
empty_rows_idx = gdf_sgi_mb[gdf_sgi_mb['RGIId_match'].apply(lambda x: len(x) == 0)].index
# drop empty rows from evaluation-gdf
gdf_sgi_mb = gdf_sgi_mb[gdf_sgi_mb['RGIId_match'].apply(lambda x: len(x) > 0)]#.reset_index()

gdf_sgi_mb.sort_values('area_km2',ascending=False,inplace=True) ## area_km2 is from SGI; AREA is from RGI.

#%%
''' -------------------
## Get GLAMOS values, use to split train/test groups
-------------------'''
sgi_list = gdf_sgi_mb['sgi-id'].unique() ## selected glaciers > 2km that have been matched to RGI
df_glamos_2km = df_glamos_mb.loc[df_glamos_mb['sgi-id'].isin(sgi_list)].copy() ## extract from all GLAMOS glaciers

print(f"glaciers in gdf_sgi_mb: {len(sgi_list)}")
print(f"glaciers in df_glamos_mb: {len(df_glamos_mb['sgi-id'].unique())}")
print(f"glaciers in df_glamos_2km: {len(df_glamos_2km['sgi-id'].unique())}")


'''## Average values per glacier, then use this to make data split'''
df_glamos_avg = df_glamos_2km.groupby(['sgi-id','name'])[['Bw','Bs','Ba','area','h_min','h_max']].mean().reset_index(inplace=False)
df_glamos_avg

## include reference year y0 and y1 for the glaciers
df_glamos_avg['y0'] = df_glamos_2km.groupby(['sgi-id','name'])['date_start'].min().reset_index(inplace=False)['date_start'].dt.year
df_glamos_avg['y1'] = df_glamos_2km.groupby(['sgi-id','name'])['date_end'].max().reset_index(inplace=False)['date_end'].dt.year


import sklearn.model_selection as skms

# Separate Aletsch training, this one needs to be in training data.
specific_sgi_id = 'B36-26'  # replace with actual ID
mask_specific = df_glamos_avg['sgi-id'] == specific_sgi_id
df_specific = df_glamos_avg[mask_specific]
df_remaining = df_glamos_avg[~mask_specific]

# Split the remaining data normally
X_train_remaining, X_test = skms.train_test_split(
    df_remaining, test_size=0.2, random_state=42, shuffle=True
)
# Combine Aletsch back into training data
X_train = pd.concat([df_specific, X_train_remaining], ignore_index=True)

## UPDATE: was error in the 'area' column since it displayed area<2km2 while there are no glaciers selected with that.
## Issue its the 'area' column which is here taken from df_glamos_mb but that is  wrong?? probably that corresponds to different assessment.
## so need to replace it with area_km2 which is from Glamos SGI file.
## BUT: not re-split train/test. Only update the values in the csv file.
## DO SAME FOR min/max elevation

## for every row in X_train and X_test, get area_km2 from gdf_sgi_mb and put in 'area' column; get h_min and h_max from gdf_sgi_mb and put in 'h_min' and 'h_max' columns
for idx, df_row in X_train.iterrows():
    sgi_id = df_row['sgi-id']
    area_km2 = gdf_sgi_mb.loc[gdf_sgi_mb['sgi-id']==sgi_id,'area_km2'].values[0]
    h_min = gdf_sgi_mb.loc[gdf_sgi_mb['sgi-id']==sgi_id,'masl_min'].values[0]
    h_max = gdf_sgi_mb.loc[gdf_sgi_mb['sgi-id']==sgi_id,'masl_max'].values[0]
    X_train.at[idx,'area'] = area_km2
    X_train.at[idx,'h_min'] = h_min
    X_train.at[idx,'h_max'] = h_max
for idx, df_row in X_test.iterrows():
    sgi_id = df_row['sgi-id']
    area_km2 = gdf_sgi_mb.loc[gdf_sgi_mb['sgi-id']==sgi_id,'area_km2'].values[0]
    h_min = gdf_sgi_mb.loc[gdf_sgi_mb['sgi-id']==sgi_id,'masl_min'].values[0]
    h_max = gdf_sgi_mb.loc[gdf_sgi_mb['sgi-id']==sgi_id,'masl_max'].values[0]
    X_test.at[idx,'area'] = area_km2
    X_test.at[idx,'h_min'] = h_min
    X_test.at[idx,'h_max'] = h_max

## plot split
fig,axs=plt.subplots(2,3,figsize=(12,6))
ax=axs[0,0]
ax.hist(X_train['Bw'], bins=20, alpha=0.7, label='Train')
ax.hist(X_test['Bw'], bins=20, alpha=0.7, label='Test')
ax.set_title('Winter Mass Balance')
ax.legend()
ax=axs[0,1]
ax.hist(X_train['Ba'], bins=20, alpha=0.7, label='Train')
ax.hist(X_test['Ba'], bins=20, alpha=0.7, label='Test')
ax.set_title('Annual Mass Balance')
ax.legend()
ax=axs[0,2]
ax.hist(X_train['Bs'], bins=20, alpha=0.7, label='Train')
ax.hist(X_test['Bs'], bins=20, alpha=0.7, label='Test')
ax.set_title('Summer Mass Balance')
ax.legend()
ax=axs[1,0]
ax.hist(X_train['area'], bins=20, alpha=0.7, label='Train')
ax.hist(X_test['area'], bins=20, alpha=0.7, label='Test')
ax.set_title('Glacier Area')
ax.legend()
ax=axs[1,1]#
ax.hist(X_train['h_min'], bins=20, alpha=0.7, label='Train (hmin)', color='#1f77b4')
ax.hist(X_test['h_min'], bins=20, alpha=0.7, label='Test (hmin)', color='#ff7f0e')
ax.hist(X_train['h_max'], bins=20, alpha=0.7, label='Train (hmax)', color="#144f7a")#,hatch='/')
ax.hist(X_test['h_max'], bins=20, alpha=0.9, label='Test (hmax)', color="#ba5a07")#,hatch='/')
ax.set_title('Min/Max Elevation')
ax.legend()
ax=axs[1,2]
ax.hist(X_train['h_max']-X_train['h_min'], bins=20, alpha=0.7, label='Train')
ax.hist(X_test['h_max']-X_test['h_min'], bins=20, alpha=0.7, label='Test')
ax.set_title('Elevation Range')
ax.legend()


fig.tight_layout()
# fig.savefig(os.path.join(homedir,'figures/development/data-glamos_split_train-test.png'), dpi=300, bbox_inches='tight')

#%% save train/test with info to csv

X_train['set'] = 'train'
X_test['set'] = 'test'
df_all = pd.concat([X_train, X_test])#, ignore_index=True)

## add refyear 
df_all['obsYears'] = df_all['y0'].astype(str) + '-' + df_all['y1'].astype(str)

## reorder columns
df_all = df_all[['sgi-id','name','set','Ba','Bw','Bs','area','h_min','h_max','obsYears']]
## round values to .3 decimals
df_all[['Ba','Bw','Bs','area']] = df_all[['Ba','Bw','Bs','area']].round(3)
df_all[['h_min','h_max']] = df_all[['h_min','h_max']].round(1)
## rename values
df_all.rename(columns={'Ba':'Ba_avg','Bw':'Bw_avg','Bs':'Bs_avg','area':'area_km2','h_min':'h_min','h_max':'h_max'}, inplace=True)
df_all

## get RGI id 
# import ast
# sgi_glamos_file = os.path.join(data_dir,'GLAMOS/sgi_glamos_rgi_matches.shp')
# gdf_glamos_sgi = gpd.read_file(sgi_glamos_file)
# # Convert list-column after loading
# gdf_glamos_sgi['RGI_match'] = gdf_glamos_sgi['RGI_match'].apply(ast.literal_eval)
gdf_glamos_sgi = gdf_sgi_mb.copy()
gdf_glamos_sgi.rename(columns={'RGIId_match':'RGI_match'}, inplace=True)

df_all = df_all.merge(gdf_glamos_sgi[['sgi-id','RGI_match']], on='sgi-id', how='left')
df_all


df_all


## to csv
if not os.path.exists('../../files/glamos_train_test_split.csv'):   
    df_all.to_csv('../../files/glamos_train_test_split.csv', index=False)
else:
    print('File already exists, not overwriting.')
