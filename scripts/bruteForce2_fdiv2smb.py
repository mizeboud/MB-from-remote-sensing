## 2D MB estimation for swiss glacier
# Brute force approach: calculate for multiple parameter set.
# Process and save output; evaluation at later stage.

# author: M Izeboud
# July 2025


# %% Imports

import xarray as xr
import rasterio as rio
import numpy as np
import os 
import glob
import geopandas as gpd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'

import myFunctions as myf

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
data_dir = '../data/' ## data directory, where input data is stored and output will be saved

## TMP:
homedir = '/Users/mizeboud/Documents/Documents_mizeboud/PostDoc/2D-SMB/'
data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
path2glacier_output = os.path.join(data_dir,'bruteForceTMP/glaciers/')

#%%
''' ################################################################

(1) Load regional (swiss) data including all glaciers to process

#################################################################### '''


'''------------------
## Glacier outlines (RGI shapefiles)
------------------'''

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs)


'''-----------------
Grab thickness & elevation data
---------------------'''

## load preprocessed reprojected file
file_grab_thickness = os.path.join(data_dir,'SwissGlacierThickness-R2020/04_IceThickness_SwissAlps/IceThickness_50m_epsg32632.tif')
da_grab = xr.open_dataarray(file_grab_thickness).isel(band=0)
da_grab.rio.crs  # EPSG 32632

file_grab_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/IceElevation_50m_epsg32632.tif') 
da_grab_elev = xr.open_dataarray(file_grab_elevation).isel(band=0)

'''-----------------
## Load hugonnet dHdt : 
---------------------'''
filename_hugo = os.path.join(data_dir,
                'GlacierElevationChange_Hugonnet2021/11_rgi60_2015-01-01_2020-01-01/',
                'dhdt/dhdt_swiss_50m_epsg32632_bilinear.tif') ## bilinearly interpolated from 100m to 50m grid
# dhdt_ID = '2015-2020_50m'
dhdt_hugo1520 = xr.open_dataarray(filename_hugo).isel(band=0).rename('dhdt')

dhdt_hugo1020 = xr.open_dataarray(os.path.join(data_dir,
                'GlacierElevationChange_Hugonnet2021/11_rgi60_2010-01-01_2020-01-01/',
                'dhdt/dhdt_swiss_50m_epsg32632_bilinear.tif')
                ).isel(band=0).rename('dhdt')
dhdt_hugo0020 = xr.open_dataarray(os.path.join(data_dir,
                'GlacierElevationChange_Hugonnet2021/11_rgi60_2000-01-01_2020-01-01/',
                'dhdt/dhdt_swiss_50m_epsg32632_bilinear.tif')
                ).isel(band=0).rename('dhdt')

#%%

''' ################################################################
Functions
#################################################################### '''

def extract_Fparam_Nlength(file, filttype=None):
    """Extract Fparam and Nlength from the filename."""
    parts = Path(file).stem.split('_')
    try:
        Fparam = float("0." + parts[-2].split('F0')[-1])
    except ValueError:
        if parts[-2] == 'F100':
            Fparam=1.0
    
    if filttype != 'f111':
        Nlength = int(parts[-1].split('N')[-1])
        return Fparam, Nlength
    if filttype == 'f111':
        Ngrad = int(parts[-1].split('-')[0].replace('Ng',''))
        Nfdiv = int(parts[-1].split('-')[1].replace('Nf',''))
        return Fparam, Ngrad, Nfdiv

def open_xrArray_assign_NFdims(file, varname='band_data', filttype=None):
    """Open a single file with xarray and rename the band, and add Nlength and Fparam as coordinates."""

    if filttype != 'f111':
        F, N = extract_Fparam_Nlength(file)
        
        with xr.open_dataset(file) as ds:
            if ds.rio.crs is None:
                ds.rio.write_crs(target_crs, inplace=True)
            ds_NF = ds.rename({'band_data':varname}).isel(band=0).drop_vars('band')
            ds_NF = ds_NF.assign_coords({'Fparam': F, 'Nlength': N}).expand_dims('Fparam').expand_dims('Nlength')
            ds_NF[varname].rio.write_nodata(np.nan, encoded=True, inplace=True) 
            return ds_NF
        
    if filttype == 'f111':
        F, Ng, Nf = extract_Fparam_Nlength(file, filttype=filttype)

        with xr.open_dataset(file) as ds:
            if ds.rio.crs is None:
                ds.rio.write_crs(target_crs, inplace=True)
            ds_NF = ds.rename({'band_data':varname}).isel(band=0).drop_vars('band')
            ds_NF = ds_NF.assign_coords({'Fparam': F, 'Ngrad': Ng, 'Nfdiv': Nf}).expand_dims('Fparam').expand_dims('Ngrad').expand_dims('Nfdiv')
            ds_NF[varname].rio.write_nodata(np.nan, encoded=True, inplace=True) 
            return ds_NF
            

def find_files_fXXX_var(path2glacier, filt_type, varname, expected_keys=None):
    """Find all files of a specific type and variable in the glacier directory.
    Args:
        path2glacier (str): Path to the glacier directory, which should contain subdirectories for different filter types, 'f000','f001', 'f110'.
        filt_type (str): Type of filter to apply ('f000', 'f001', 'f110').
        varname (str): Name of the variable to search for in the files.
    Returns: list of files matching the criteria."""

    files_fXXX_var = glob.glob(os.path.join(path2glacier,f'{filt_type}/*{varname}*.tif'))
    if not files_fXXX_var:
        print(f"No '{varname}' file found in {path2glacier}/{filt_type}/.")
        raise FileNotFoundError(f"No '{varname}' file found in {path2glacier}/{filt_type}/.")
    file0 = files_fXXX_var[0]
    if expected_keys is not None:
        actual_keys = {
            extract_Fparam_Nlength(file, filttype=filt_type)
            for file in files_fXXX_var
        }
        missing_keys = set(expected_keys) - actual_keys
        unexpected_keys = actual_keys - set(expected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"Unexpected {filt_type} file set for {Path(file0).stem.split('_F')[0]}. "
                f"Missing: {sorted(missing_keys)}; "
                f"unexpected: {sorted(unexpected_keys)}."
            )

    # print(f"Loading {len(files_fXXX_var)} files for  {Path(file0).stem.split('_F')[0]}*.")

    files_fXXX_var.sort()  # Sort the files to ensure consistent order
    return files_fXXX_var

def load_combined_dataset_variable(filelist, filt_type, varname):
    '''Load and combine datasets from a list of files based on filter type and variable name.
    Args:
        filelist (list): List of file paths to be combined.
        filt_type (str): Type of filter applied to the files (e.g., 'f000', 'f001', 'f110').
        varname (str): Name of the variable to be assigned to the dataset. (fluxdiv; mb)
    Returns:
        ds_combined (xarray.Dataset): Combined dataset with dimensions for Fparam and Nlength
    '''
    
    from collections import defaultdict
    
    if filt_type != 'f111':
        grouped = defaultdict(dict) #  to hold grouped files by N and F

        ## need to build nested list (F,N) for xarray combine_nested
        for file in filelist:
            # get F and N values from filenames
            file_Fparam, file_Nlength = extract_Fparam_Nlength(file, filttype=filt_type)
            ## group filenames into a nested list based on Fparam and Nlength
            grouped[file_Fparam][file_Nlength] = file
            # build nested list sorted by Fparam then Nlength
            filelist_nested = [
                [grouped[Fparam][Nlength] for Nlength in sorted(grouped[Fparam].keys())]
                for Fparam in sorted(grouped.keys())
            ]

        ds_nestlist = [
            [open_xrArray_assign_NFdims(file,varname=f'{varname}_{filt_type}', filttype=filt_type) for file in files_nested_row]  # Open all files in the first group
            for files_nested_row in filelist_nested # loop over each row of nested files
        ]

        ds_combined = xr.combine_nested(ds_nestlist, concat_dim=['Fparam','Nlength'])
        # print('.. combined all files into dataset:', ds_combined)

    if filt_type == 'f111':
        ## need to build nested list in two steps to get (F,Ng,Nf) for xarray combine_nested
        ds_list_Fnest = []
        f_values = sorted({
            extract_Fparam_Nlength(file, filttype='f111')[0]
            for file in filelist
        })
        for f_value in f_values:
            group_F_files = [
                file for file in filelist
                if extract_Fparam_Nlength(file, filttype='f111')[0] == f_value
            ]
            grouped_F = defaultdict(dict) #  to hold grouped files by N and F
            
            for file in group_F_files:
                file_Fparam, file_Ngrad, file_Nfdiv = extract_Fparam_Nlength(file, filttype=filt_type)
                grouped_F[file_Ngrad][file_Nfdiv] = file

            # build nested list sorted by Fparam then Nlength
            filelist_nested = [
                [grouped_F[Ngrad][Nfdiv] for Nfdiv in sorted(grouped_F[Ngrad].keys())]
                for Ngrad in sorted(grouped_F.keys())
            ]
        
            ds_nestlist = [
                [open_xrArray_assign_NFdims(file,varname=f'{varname}_{filt_type}', filttype=filt_type) for file in files_nested_row]  # Open all files in the first group
                for files_nested_row in filelist_nested # loop over each row of nested files
            ]


            ds_combined_NgNf = xr.combine_nested(ds_nestlist, concat_dim=['Ngrad','Nfdiv'])
            # print(ds_combined_NgNf.Fparam.values, ds_combined_NgNf.dims)
            ds_list_Fnest.append(ds_combined_NgNf)
        
        ds_combined = xr.combine_nested(ds_list_Fnest, concat_dim=['Fparam'])


    return ds_combined 

def read_and_combine_glacier_data(path2glacier, varname, expected_keys):
    
    ''' --------------------------------------------------------------
    F000: Combine all fluxdiv files for F000
    ------------------------------------------------------------------'''
    filelist_f000 = find_files_fXXX_var(
        path2glacier, 'f000', varname, expected_keys=expected_keys['f000'])
    ds_f000 = load_combined_dataset_variable(filelist_f000, 'f000', varname)
    # print('.. combined all files into dataset:', ds_f000.sizes)#, ds_f000.data_vars)
    # print(ds_f000.sizes) 

    ''' --------------------------------------------------------------
    F001: Combine all fluxdiv files for F001
    ------------------------------------------------------------------'''
    
    filelist_f001 = find_files_fXXX_var(
        path2glacier, 'f001', varname,
        expected_keys=expected_keys['f001'])
    ds_f001 = load_combined_dataset_variable(filelist_f001, 'f001', varname)
    # print('.. combined all files into dataset:', ds_f001.sizes)#, ds_f001.variables)

    ''' --------------------------------------------------------------
    F110: Combine all fluxdiv files for F110
    ------------------------------------------------------------------'''
    filelist_f110 = find_files_fXXX_var(
        path2glacier, 'f110', varname,
        expected_keys=expected_keys['f110'])
    ds_f110 = load_combined_dataset_variable(filelist_f110, 'f110', varname)
    # print('.. combined all files into dataset:', ds_f110.sizes)#, ds_f110.variables)


    
    ''' --------------------------------------------------------------
    Combine all in a big dataset
    ------------------------------------------------------------------'''

    ds_glacier = xr.merge([ds_f000,ds_f001,ds_f110],
                          join='outer', ## Nlength is not the same for f000 vs f001/f110, so set join=outer to keep all values and fill missing with NaN
                          compat='no_conflicts'
                          ) # merge datasets on common dimensions (x,y,Nlength,Fparam) and fill missing values with NaN
    # print('.. combined all f### variables into dataset:', ds_glacier.sizes)#
    ds_glacier.attrs['description'] = f'Distributed estimates of {varname} from mass continuity inversion'
    ds_glacier.attrs['units'] = 'm.i.e./yr'
    ds_glacier.attrs['CRS'] = 'EPSG:32632'
    ds_glacier.attrs['f000'] = 'Filter approach: No spatial smoothing; gradients calculated on central-difference scheme'
    ds_glacier.attrs['f001'] = 'Filter approach: Spatial smoothing with exponential filter, applied to flux divergence field'
    ds_glacier.attrs['f110'] = 'Filter approach: Spatial smoothing with exponential filter, applied to horizontal gradients of ice thickness and velocity components'
    ds_glacier.attrs['Fparam'] = 'Ratio used to convert surface velocity to depth-averaged velocity'
    ds_glacier.attrs['Nlength'] = 'Decay length N of exponential filter weights, impelmented as N x Ice Thickness'
    
    # print(ds_glacier)
    return ds_glacier


def expected_file_keys(param_grid):
    """Return unique filename parameter keys for each filter type.
    f000: (N=0, Fparam)
    f001: (N, Fparam)
    f110: (N, Fparam)
    f111: (Ngrad, Nfdiv, Fparam)
    """
    return {
        'f000': {(round(fparam, 2), 0) for _, _, fparam in param_grid},
        'f001': {
            (round(fparam, 2), lscale_fdiv)
            for _, lscale_fdiv, fparam in param_grid
        },
        'f110': {
            (round(fparam, 2), lscale_gradients)
            for lscale_gradients, _, fparam in param_grid
        },
        'f111': {
            (round(fparam, 2), lscale_gradients, lscale_fdiv)
            for lscale_gradients, lscale_fdiv, fparam in param_grid
            if lscale_gradients <= 5 and lscale_fdiv <= 5
        },
    }

#%%

'''-----------------
## Set up parameter grid  and choose workflow
---------------------'''

savenc = True

param_grid, max_ksize_pxs = myf.make_parameter_grid()
expected_keys = expected_file_keys(param_grid) # F, Ngrad, Nfdiv for f111; F, N for f000, f001, f110


''' --------------
Get list of finished glaciers
------------------'''

## get list of finished glaciers
list_RGI_dirs = os.listdir(path2glacier_output)
list_RGI_dirs.sort()
list_RGI_dirs = [d for d in list_RGI_dirs if d.startswith('RGI')] # remove .DS_store etc from directory list
# list_RGI_dirs = [d for d in list_RGI_dirs if d in rgi_list]  # Filter directories to only those that are in the RGI list
print('Number of glaciers that were processed: ', len(list_RGI_dirs))

#%%

''' ################################################################
Combine output for each glacier
#################################################################### '''
# for glacier_rgi in list_RGI_dirs[:20]:
for glacier_idx in tqdm(range(0,len(list_RGI_dirs)), desc="Combining glacier output files"): 
# for glacier_idx in range(51,len(list_RGI_dirs)):
    glacier_rgi = list_RGI_dirs[glacier_idx]
    
    path2glacier = os.path.join(path2glacier_output, glacier_rgi)

    ''' --------------------------------------------------------------
    Load, combine and save data
    ------------------------------------------------------------------'''
    # MB dir
    path2saveMB = os.path.join(path2glacier, 'mb')
    os.makedirs(path2saveMB, exist_ok=True) # create base directory if it doesn't exist

    ## check if file already exists -- continue to next if so
    filen_nc_fdiv = os.path.join(path2glacier, f'{glacier_rgi}_fluxdiv_paramgrid.nc' )
    filen_nc_mb = os.path.join(path2saveMB, f'{glacier_rgi}_mb_2000-2020_paramgrid.nc' ) ## check for 1 period; assuming that if output of 1 is availalbe, all are.

    if os.path.exists(filen_nc_fdiv) and os.path.exists(filen_nc_mb):
        print(f'{glacier_rgi} output exists, continue')
        continue


    ''' --------------------------------------------------------------
    F000, f001, f110: Combine all fluxdiv files; save to netcdf. Dimensiosn (x,y,N,F)
    ------------------------------------------------------------------'''
    ### combine files for f000, f001, f110 (netcdf with (x,y,N,F) dimensions)
    ds_glacier_fdiv = read_and_combine_glacier_data(path2glacier, 'fluxdiv', 
                                                    expected_keys=expected_keys,
                                                    ) 
    ## save data
    if not os.path.exists(filen_nc_fdiv):
        if savenc:
            ds_glacier_fdiv.to_netcdf(filen_nc_fdiv )  

    ''' --------------------------------------------------------------
    F111: Combine all fluxdiv files for F111; save as separate netcdf since dimensions are different than other approaches (x,y, Ngrad, Nfdiv,F)
    ------------------------------------------------------------------'''
    filelist_f111 = find_files_fXXX_var(
        path2glacier,
        'f111',
        'fluxdiv',
        expected_keys=expected_keys['f111'],
    )
    ds_f111 = load_combined_dataset_variable(filelist_f111, filt_type='f111', varname='fluxdiv')
    ds_f111.attrs['description'] = f'Distributed estimates of fluxdiv from mass continuity inversion'
    ds_f111.attrs['units'] = 'm.i.e./yr'
    ds_f111.attrs['CRS'] = 'EPSG:32632'
    ds_f111.attrs['f111'] = 'Filter approach: Spatial smoothing with exponential filter, applied to horizontal gradients of ice thickness and velocity components'
    ds_f111.attrs['Fparam'] = 'Ratio used to convert surface velocity to depth-averaged velocity'
    ds_f111.attrs['Ngrad'] = 'Decay length N of exponential filter weights, impelmented as N x Ice Thickness on smoothing gradients'
    ds_f111.attrs['Nfdiv'] = 'Decay length N of exponential filter weights, impelmented as N x Ice Thickness on smoothing flux divergence field'
    ## save data
    if not os.path.exists(filen_nc_fdiv.replace('_fluxdiv_paramgrid.nc', '_fluxdiv-f111_paramgrid.nc')):
        if savenc:
            ds_f111.to_netcdf(filen_nc_fdiv.replace('_fluxdiv_paramgrid.nc', '_fluxdiv-f111_paramgrid.nc') )  


    ''' --------------------------------------------------------------
    Calculate MB
    ------------------------------------------------------------------'''
    gdf_current_glacier = gdf_swiss_large.loc[gdf_swiss_large['RGIId']==glacier_rgi]
    glacier_name = gdf_current_glacier['Name'].item()

    if not os.path.exists(filen_nc_mb):
        
        ''' ################################################################
        Calculate MB for each dhdt period
        #################################################################### '''

        for dhdt_hugo, dhdt_label in zip([dhdt_hugo0020,dhdt_hugo1020, dhdt_hugo1520],
                                           ['2000-2020', '2010-2020', '2015-2020']):
            da_dhdt_period = dhdt_hugo.rio.clip(gdf_current_glacier.geometry, drop=True,all_touched=True)

            # Density conversion from m.i.e./yr to kg/m2/yr.
            ''' ----------------------------------------------
            Density conversion: convert from m.i.e./yr to kg/yr using ice density
            MB = rho/rhowater * dhdt + rho/rhowater * fluxdiv
            1. flux divergence density = 900 kg/m3; so scale values by 0.9 (rho_flux / rho_water)
            2. dhdt density accumulation area = 600 kg/m3; --> places where both dhdt & fdiv are positive
            3. dhdt density ablation area = 900 kg/m3 --> places where both dhdt & fdiv are negative
            4. other pixels: use 850
            -------------------------------------------------- '''


            '''## ---- apply mb conversion'''
            # mb_mwe, dhdt_mwe, fdiv_mwe = myf.compute_mb_mwe(da_fdiv, da_dhdt)
            mb_mwe_fXXX_period, _, _ = myf.compute_mb_mwe(ds_glacier_fdiv, da_dhdt_period)
            mb_mwe_f111_period, _, _ = myf.compute_mb_mwe(ds_f111, da_dhdt_period)

            ## update attributes
            mb_mwe_fXXX_period.attrs['description'] = f'Distributed estimates of MB from mass continuity inversion, using {dhdt_label} dhdt'
            mb_mwe_fXXX_period.attrs['units'] = 'm.w.e./yr'
            mb_mwe_fXXX_period.attrs['long_name'] = 'mass balance'
            mb_mwe_fXXX_period = mb_mwe_fXXX_period.rename({'fluxdiv_f000': 'mb_f000', 'fluxdiv_f001': 'mb_f001', 'fluxdiv_f110': 'mb_f110'})
            for var in mb_mwe_fXXX_period.data_vars: ## copy attr to each 
                mb_mwe_fXXX_period[var].attrs = mb_mwe_fXXX_period.attrs

            mb_mwe_f111_period.attrs['description'] = f'Distributed estimates of MB from mass continuity inversion, using {dhdt_label} dhdt and f111 fluxdiv'
            mb_mwe_f111_period.attrs['units'] = 'm.w.e./yr'
            mb_mwe_f111_period.attrs['long_name'] = 'mass balance'
            mb_mwe_f111_period = mb_mwe_f111_period.rename({'fluxdiv_f111': 'mb_f111'})
            for var in mb_mwe_f111_period.data_vars: ## copy attr to each 
                mb_mwe_f111_period[var].attrs = mb_mwe_f111_period.attrs
            
            ## save converted MB
            filen_mb_fXXX_kg = os.path.join(path2saveMB, f'{glacier_rgi}_mb_{dhdt_label}_paramgrid.nc')
            file_mb_f111_kg = os.path.join(path2saveMB, f'{glacier_rgi}_mb-f111_{dhdt_label}_paramgrid.nc')
            if not os.path.exists(filen_mb_fXXX_kg):
                if savenc:
                    mb_mwe_fXXX_period.to_netcdf(filen_mb_fXXX_kg)
                    mb_mwe_f111_period.to_netcdf(file_mb_f111_kg)
            

# %%
