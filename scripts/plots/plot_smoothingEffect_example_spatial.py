
## BRUTE FORCE for single glacier
#%%
import xarray as xr
import rioxarray as rioxr
import rasterio as rio
import numpy as np
import os 
import ast
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib
from shapely.geometry import Polygon


# ## need to specify workdir as I have files in iCloud and Spyder doesnt understand that.
# homedir = '/Users/mizeboud/Documents/Documents_mizeboud/PostDoc/2D-SMB/'
# data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
# workdir = os.path.join(homedir, 'code/SMB-from-remote-sensing/scripts/')
# os.chdir(workdir)
# import myFunctions as myf

# Add the directory containing functions.py to path
import sys
sys.path.append(os.path.dirname(os.path.abspath('../scripts')))
import myFunctions as myf

### General settings / paths
target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution), all files are processed in this CRS.
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes

data_dir = '/Users/mizeboud/Documents/Data_iCloud/SMB2D/'
# data_dir = '../data'

path2glacier_output = os.path.join(data_dir,'bruteForceTMP/glaciers/')
path2saveRSME = os.path.join(data_dir, 'bruteForceTMP/rmse2glamos/')


path2save_figure = '../../figures/plot_smoothEffect/'
os.makedirs(path2save_figure, exist_ok=True)


# %% Read data 


'''------------------
## Glacier outlines (RGI shapefiles)
------------------'''

gdf_swiss_large = myf.load_rgi_outlines_swiss(filepath = os.path.join(data_dir,'RGI/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp'),
                                          area_km2=2, target_crs=target_crs)


## load preprocessed reprojected file

file_grab_thickness = os.path.join(data_dir,'SwissGlacierThickness-R2020/04_IceThickness_SwissAlps/IceThickness_50m_epsg32632.tif')
da_grab = xr.open_dataarray(file_grab_thickness).isel(band=0)

file_grab_elevation = os.path.join(data_dir,'SwissGlacierThickness-R2020/08_SurfaceElevation_SwissAlps/IceElevation_50m_epsg32632.tif') 
da_grab_elev = xr.open_dataarray(file_grab_elevation).isel(band=0)

#%% FUnction

''' ######################################
Function
##########################################'''


def plot_glacier_data(data, minmax,
                    #   glacier_sgiid,glacier_name, 
                      rgi_outlines, sgi_outlines=None, 
                      cmap='RdBu_r',
                      plot_cbar=True, plot_scalebar=False ,clabel='',title='',ax=None
                      ):
    from matplotlib_scalebar.scalebar import ScaleBar
    if len(minmax)==1:
        vmin = -minmax[0]
        vmax= minmax[0]
    else:
        vmin = minmax[0]
        vmax = minmax[1]
    if ax is None:
        fig,ax=plt.subplots(1,figsize=(10,6))

    if plot_cbar:
        data.plot.imshow(ax=ax,cmap=cmap,vmin=vmin, vmax=vmax, 
                        cbar_kwargs={'fraction':0.046,'pad':0.04,
                                    'label':clabel})  
    else:
        data.plot.imshow(ax=ax,cmap=cmap,vmin=vmin, vmax=vmax, add_colorbar=False)
    # ax.set_title(f'{glacier_sgiid} {glacier_name}')
    ax.set_title(title)
    ax.set_aspect('equal')

    if plot_scalebar:
        scalebar=ScaleBar(dx=1, # size of pixel
                          units='m',
                          location='lower left',
                          scale_loc='top',
                          box_alpha=0.5,
                          )
        ax.add_artist(scalebar)
        ax.set_axis_off()
    ## add outlines
    # for ax in axs.flatten():
    if sgi_outlines is not None:
        sgi_outlines.boundary.plot(ax=ax,label='GLAMOS SGI lilne')
    rgi_outlines.boundary.plot(ax=ax,color='k', label='RGI lines')
    
    # plt.tight_layout()
    return ax



#%% 

''' --- PLOT SETTINGS ---- '''
f000_inclusive = True # include f000 with N=0 in gif (if True) or keep as separate (if False)
static_F = True # if True, plot only one F value (F=0.8) and vary N; if False, vary F and keep N=0
fparam = 0.80
fparam_str = f'{fparam:.2f}'.replace('.','')
''' -----------------------'''


savefig = False


''' ######################################
Select glacier, get data
##########################################'''

print(len(gdf_swiss_large))


'''######################################
## Select glacier by RGI or name
###################################### '''
rgi_to_plot = ['RGI60-11.02773', 'RGI60-11.01450'] # Aletsch & Findel glacier

for glacier_rgiid_select in rgi_to_plot:
    # glacier_name = 'Aletsch'
    # glacier_rgiid_select = 'RGI60-11.01450' ## Aletsch

    gdf_current_glacier = gdf_swiss_large.loc[gdf_swiss_large['RGIId'] == glacier_rgiid_select] ## SELECT GLACIER Aletsch, RGI60-11.01450

    gdf_current_glacier
    gdf_glacier_rgi = gdf_current_glacier.copy()
    glacier_rgiid = glacier_rgiid_select


    ''' ######################################
    Load brute force output 
    ##########################################'''

    ds_fdiv_paramgrid = xr.open_dataset(os.path.join(path2glacier_output, glacier_rgiid_select, f'{glacier_rgiid_select}_fluxdiv_paramgrid.nc'))
    # ds_fdiv_paramgrid_f111 = xr.open_dataset(os.path.join(path2glacier_output, glacier_rgiid_select, f'{glacier_rgiid_select}_fluxdiv-f111_paramgrid.nc'))


    ## select slices to plot

    da_f000 = ds_fdiv_paramgrid['fluxdiv_f000'].sel(Nlength=0, Fparam=fparam) ## select f000 with N=0 and fparam=0.75 (default value)
    if static_F:
        da_f001 = ds_fdiv_paramgrid['fluxdiv_f001'].sel(Fparam=fparam,  ## select f001 with  fparam=0.75 (default value)
                                                        Nlength=slice(1, None) ## and all N>0 (N=0 has nan values for f001)
                                                        )
        
        da_f110 = ds_fdiv_paramgrid['fluxdiv_f110'].sel(Fparam=fparam,  ## select f110 with  fparam=0.75 (default value)
                                                        Nlength=slice(1, None) ## and all N>0 (N=0 has nan values for f110)
                                                        )

    ''' --------------------
    ### PLOT and save figures: plot fluxdiv and smb 3-panels
    --------------------'''

    # Save figures: plot fluxdiv and smb 3-panels
    cmap_fdiv = matplotlib.cm.RdBu_r
    # cmap_fdiv.set_bad('#F7CF59',1.)
    cmap_fdiv.set_bad("#E2E2E2",1.) # light gray
    cmap_smb = matplotlib.cm.RdBu
    # cmap_smb.set_bad('#F7CF59',1.)
    cmap_smb.set_bad("#E2E2E2",1.) # light gray


    ''' ---- SELECT WHICH DATA TO PLOT ---- '''
    filttype = 'f001'; smoothtype = 'filter flux.div';
    da_plot_fXXX = da_f001.copy()

    # filttype = 'f110'; smoothtype = 'filter gradients';
    # da_plot_fXXX = da_f110.copy()

    fig,axs = plt.subplots(2,6,figsize=(16,7))
    c,r = 0,0
    for N in ds_fdiv_paramgrid.Nlength.values:
        if N == 0:
            data = da_f000
        else:
            data = da_plot_fXXX.sel(Nlength=N) 

        # fig,ax=plt.subplots(1,figsize=(6,7))
        ax= axs[r,c]
        smoothtype = smoothtype + f' ({filttype})'
        cmap=cmap_fdiv
        minmax  = [8]
        ax = plot_glacier_data(data, minmax,
                            # glacier_sgiid,glacier_name, 
                            rgi_outlines=gdf_glacier_rgi, sgi_outlines=None, 
                            cmap=cmap,
                            plot_scalebar=False, 
                            plot_cbar = False, clabel='m.i.e./yr',
                            title=f'',#{smoothtype}: F={fparam_str}, N={N}',
                            ax=ax#s[col]
                            )
        ax.set_axis_off()
        ## add text box with N value
        ax.text(0.02, 0.1, f'N={N}', transform=ax.transAxes, fontsize=18, 
        # ax.text(0.02, 0.98, f'N={N}', transform=ax.transAxes, fontsize=18, 
                ## shade box background
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                verticalalignment='top')

        c += 1
        if c == 6:
            c = 1
            r += 1
    axs[1,0].set_axis_off()

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # [left, bottom, width, height]
    norm = matplotlib.colors.Normalize(vmin=-minmax[0], vmax=minmax[0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('flux divergence (m.i.e./yr)', fontsize=14)
    fig.suptitle(f'{glacier_rgiid}: smoothing effect on filtering only flux divergence ({filttype})', fontsize=16, position=(0.5, 0.92))
    figname = f'{glacier_rgiid}_fluxdiv_{filttype}_smoothingEffect'
    print(figname)
    if savefig:
        # fig.savefig(os.path.join(path2save_figure,figname+'.pdf'),bbox_inches='tight',transparent=True)
        fig.savefig(os.path.join(path2save_figure,figname+'.png'),bbox_inches='tight',transparent=True, dpi=300)
        # plt.close()
    # break

# %%
