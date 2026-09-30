import itertools

import numpy as np
import rioxarray
import rasterio as rio
import xarray as xr

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import box
import time 
from numba import njit, prange 

from sklearn.linear_model import LinearRegression
import warnings

from scipy.interpolate import griddata

def load_rgi_outlines_swiss(filepath, 
                            area_km2 = 2, 
                            target_crs='EPSG:32632', 
                            box_bounds = [ 243071.95534342, 5049658.05308324,  634818.83410692, 5331921.25252423] ):
    """Load RGI outlines clipped to Swiss bounds, merged with Claridenfirn RGI parts."""
    gl_outline_shp_rgi11 = gpd.read_file(filepath)
    gl_outline_shp_rgi11 = gl_outline_shp_rgi11.to_crs(target_crs)

    ## create polygon of swiss bounds 
    # swiss_bounds = [ 243071.95534342, 5049658.05308324,  634818.83410692, 5331921.25252423] # in EPSG:32632; including small buffer
    swiss_bounds_poly = gpd.GeoDataFrame(index=[0], crs=target_crs, geometry=[Polygon.from_bounds(*box_bounds)])  

    ''' ALL SWISS glaciers; use RGI outliens; only larger than area threshold '''
    rgi_outlines_swiss = gpd.sjoin(gl_outline_shp_rgi11, swiss_bounds_poly).drop('index_right',axis=1) # intersection using geopandas SJOIN(left, right) clips LEFT to the RIGHT geom
    gdf_swiss_large = rgi_outlines_swiss.loc[rgi_outlines_swiss['Area']>area_km2]

    '''## adjust claridenfirn glacier RGI
    - Claridenfirn in SGI outlines is composed of multiple RGI glaciers
    - I want to process these RGIs as one --> merge
    '''
    rgi_claridenfirn = ['RGI60-11.00817', 'RGI60-11.00819', 'RGI60-11.00843', 'RGI60-11.00878']
    ## merge the 4 RGI regions
    gdf_claridenfirn = rgi_outlines_swiss.loc[rgi_outlines_swiss['RGIId'].isin(rgi_claridenfirn)].dissolve() # merges all geometries into 1
    gdf_claridenfirn['RGIId'] = 'RGI60-11.008merged' # make up new RGIId
    gdf_claridenfirn['Name'] = 'Claridenfirn'
    ## also update other fields
    gdf_claridenfirn['Area'] = gdf_claridenfirn['Area'].sum()
    gdf_claridenfirn['Zmin'] = gdf_claridenfirn['Zmin'].min()
    gdf_claridenfirn['Zmax'] = gdf_claridenfirn['Zmax'].max()
    gdf_claridenfirn[['Zmed','Slope','Surging','Aspect','Lmax','Linkages']] = np.nan

    ## add merged glacier to gdf_swiss_large
    gdf_swiss_large = pd.concat([gdf_swiss_large, gdf_claridenfirn], ignore_index=True)

    print(f'{len(gdf_swiss_large)} RGI glaciers with Area > {area_km2} km2 filtered (including Claridenfirn merged)')
    return gdf_swiss_large

def reproject_match_grid( ref_img_da, img_da , resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan):
    ''' Match xarray grid of different spatial resolutions. Input should be dataArray'''

    # Expected order: ('time', 'y', 'x')
    dims = img_da.dims
    if 'time' in dims:
        ref_img_da = ref_img_da.transpose('time','y','x') # CRS is alreadyy written .rio.write_crs(3031, inplace=True)
        img_da = img_da.transpose('time','y','x')
    
    # -- reproject (even though same crs) and match grid (extent, resolution and projection)
    img_repr_match = img_da.rio.reproject_match(ref_img_da,resampling=resample_method,nodata=nodata_value) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

    # advised to update coords to make the coordinates the exact same due to tiny differences in the coordinate values due to floating precision
    img_repr_match = img_repr_match.assign_coords({
        "y": ref_img_da.y,
        "x": ref_img_da.x,
    })
    
    return img_repr_match.transpose(*dims) # transpose dimension order back to original


# calculate image boundaries
def img_bound_gpd(img_da): # input: xarray dataArray
    '''Create shapely polygon based on boundaries of xr.DataArray'''
    polygon_geom = Polygon.from_bounds(*img_da.rio.bounds())
    polygon = gpd.GeoDataFrame(index=[0], crs=img_da.rio.crs, geometry=[polygon_geom])  
    return polygon


def calc_spatial_gradient(da, length_scale_px=1 , dx=None ):
    '''Calculate spatial gradient based on x and y grids, allowing variation for lenght-scale to be taken into account
    The gradient is computed using second order accurate central differences in the interior points 
    and either first or second order accurate one-sides (forward or backwards) differences at the boundaries


    Input
    -----
    Should be xr.DataArray or xr.DataSet
    length_scale : Integer, it represnts the number of pixels to take as length scale. 
    dx           : Integer, grid resolution (m)

    Output
    ------
    d/dx, d/dy
    
    '''
    

    # Infer spatial resolution of grid
    if dx is None:
        dx = int(da.rio.resolution()[0])
        dy = int(da.rio.resolution()[1])
        if np.abs(dx) != np.abs(dy):
            raise ValueError("x and y resolution are not the same; {} and {} -- code update required".format(np.abs(dx), np.abs(dy) ))

    
    '''# optie: use coords'''
    if length_scale_px == 1:
        da_gradient = np.gradient(da,
                    da.y, da.x, # Use Arrays to specify the coordinates of thhe values along each dimension of F.     
                    edge_order= 2, # {1, 2}
        ) # output is np array so need to coonvert back to xarrya
    else:
        ##  Length scale settings 
        px_dist  = length_scale_px   # number of pixels to shift
        # res = np.abs(dx)*px_dist     # distance between points of gradient

        '''## Calculate gradients'''
        # 2nd oorder accuratae, central-difference scheme, with 1st or 2nd order at bounds
        da_gradient = np.gradient(da,
                    px_dist, # single scalar to specify a sample distance for all dimensions. 
                    edge_order= 2, # {1, 2}
        ) # output is np array so need to coonvert back to xarrya
        
    
    ## dimension order: expected to be (y,x) so [0] is dy, but add check
    d0 = da.dims[0]; d1 = da.dims[1] 
    if not d0 == 'y': raise ValueError(f'Expected dimension order (y,x) but got {da.dims}')
    if not d1 == 'x': raise ValueError(f'Expected dimension order (y,x) but got {da.dims}')
    
    # put back in xarray
    da_dx = da.copy(data=da_gradient[1]).drop_attrs().rename('dx') 
    da_dy = da.copy(data=da_gradient[0]).drop_attrs().rename('dy')
    
    
    return da_dx, da_dy # da_gradient # dudx, dudy

def wrap_calc_centraldiff_gradients(da_thickness, da_vx, da_vy, 
                                    gdf_current_glacier, target_crs,
                                    extrapolate_method='nanmask'):

    '''-----------------
    ## 0. GRADIENTS: calculate simple central-diff gradients
    (1) extrapolate at edges of glacier (to avoid NaNs bleeding into glacier bounds)
    (2) calculate spatial gradients on standard central-diff scheme (delta = 1 px)
    (3) re-update glacier bounds mask to remove unnecessary extrapolation
    ---------------------'''

    
    if extrapolate_method is not None: 
        # print('extrapolation with ' + extrapolate_method)
        ''' Extrapolate at edges of glacier'''
        da_thickness1 = extrapolate_nan_2d(da_thickness)
        da_vx1 = extrapolate_nan_2d(da_vx)
        da_vy1 = extrapolate_nan_2d(da_vy)
            
        ''' Calculate spatial gradients without taking length scale into account yet  '''
        dHdx, dHdy = calc_spatial_gradient(da_thickness1, length_scale_px=1 )
        dudx, dudy = calc_spatial_gradient(da_vx1, length_scale_px=1 )
        dvdx, dvdy = calc_spatial_gradient(da_vy1, length_scale_px=1 )
    else: 
        ''' Calculate spatial gradients without taking length scale into account yet  '''
        dHdx, dHdy = calc_spatial_gradient(da_thickness, length_scale_px=1 )
        dudx, dudy = calc_spatial_gradient(da_vx, length_scale_px=1 )
        dvdx, dvdy = calc_spatial_gradient(da_vy, length_scale_px=1 )
    
    if extrapolate_method == 'clip':
        '''## Clip gradients to glacier outlines (use union, to have most data included)'''
        dHdx = dHdx.rio.write_crs(target_crs).rio.clip(gdf_current_glacier.geometry,drop=True,all_touched=True)
        dHdy = dHdy.rio.write_crs(target_crs).rio.clip(gdf_current_glacier.geometry,drop=True,all_touched=True)
        dudx = dudx.rio.write_crs(target_crs).rio.clip(gdf_current_glacier.geometry,drop=True,all_touched=True)
        dudy = dudy.rio.write_crs(target_crs).rio.clip(gdf_current_glacier.geometry,drop=True,all_touched=True)
        dvdx = dvdx.rio.write_crs(target_crs).rio.clip(gdf_current_glacier.geometry,drop=True,all_touched=True)
        dvdy = dvdy.rio.write_crs(target_crs).rio.clip(gdf_current_glacier.geometry,drop=True,all_touched=True)
    
    elif extrapolate_method == 'nanmask':
        dHdx = dHdx.where(~np.isnan(da_thickness) )
        dHdy = dHdy.where(~np.isnan(da_thickness) )
        dudx = dudx.where(~np.isnan(da_vx) )
        dudy = dudy.where(~np.isnan(da_vx) )
        dvdx = dvdx.where(~np.isnan(da_vy) )
        dvdy = dvdy.where(~np.isnan(da_vy) )

    return dHdx, dHdy, dudx, dudy, dvdx, dvdy
    

def calc_flux_div(thickness, dHdx, dHdy, u_surf, dudx, dudy, v_surf, dvdx, dvdy, fparameter=0.8):
    '''Calculate flux divergence (m.i.e. /yr)
    
    Input
    -----
    thickness, dHdx, dHdy   :   local ice thickness and its spatial gradients
    u_surf, dudx, dudy      :   local surface velocity (x-ax) component and spatial gradients of the surface velocity
    v_surf, dvdx, dvdy      :   local surface velocity (y-ax) component and spatial gradients of the surface velocity
    fparameter              :   parameter to account for basal sliding in the SIA equation. F ranges from 0.8 (no sliding) to 1 (fully sliding)

    Output
    ------
    flux_div   :   Flux divergence
    
    '''
    term1 = fparameter * u_surf * dHdx;
    term2 = fparameter * v_surf * dHdy;
    term3 = fparameter * thickness * dudx;
    term4 = fparameter * thickness * dvdy;

    flux_div = term1 + term2 + term3 + term4;
    return flux_div 

def scale_fluxdiv_uniform(fluxdiv_to_scale, fluxdiv_ref):
    '''## fratio should be absolute, otherwise sudden sign switch can happen
    ## also, fdiv.sum should not be between 0-1 because then values explode.'''
    if np.abs(fluxdiv_to_scale.sum()) < 1: # value between 0-1
        # set fdiv_sum to 1 ? need to think on what it should be
        fdiv_ratio = -999
        ## set fdiv_ratio to -999 so I can catch it later in postprocessing; but keep fluxdiv_scaled as it is without scaling so I can see if it is actually problematic.
        fluxdiv_scaled = fluxdiv_to_scale.copy() # fluxdiv_to_scale.where(np.isnan(fluxdiv_to_scale),-999)
    else: # if sum >1 
        fdiv_ratio = np.abs( fluxdiv_ref.sum() / fluxdiv_to_scale.sum()  ).item()
        fluxdiv_scaled = fluxdiv_to_scale * fdiv_ratio
    return fluxdiv_scaled, fdiv_ratio


def compute_mb_mwe(da_fdiv, da_dhdt):
    ''' Combine one flux-divergence field and (one) dhdt field into MB (m.w.e/yr), i.e. also apply density conversion.
    The accumulation/ablation density mask is derived from the (fdiv, dhdt) pair,
    Density conversion: convert from m.i.e./yr to m w.e./yr using ice density
        MB = rho/rho_water * dhdt + rho/rho_water * fluxdiv
        1. flux divergence density = 900 kg/m3; so scale values by 0.9 (rho_flux / rho_water)
        2. dhdt density accumulation area = 600 kg/m3; --> places where both dhdt & fdiv are positive
        3. dhdt density ablation area = 900 kg/m3 --> places where both dhdt & fdiv are negative
        4. other pixels: use 850 kg/m3
    '''
    ## set up grid with conversion factor
    density_conversion =    xr.where((da_fdiv > 0) & (da_dhdt > 0), 0.600,         ## accumulation area
                            xr.where((da_fdiv < 0) & (da_dhdt < 0), 0.900, 0.850)) ## ablation area, else 850
    density_conversion = density_conversion.where(~np.isnan(da_fdiv), np.nan)
    ## apply conversion
    dhdt_mwe = da_dhdt * density_conversion
    fdiv_mwe = da_fdiv * 0.9
    ## mass balance in m w.e./yr
    mb_mwe = dhdt_mwe + fdiv_mwe
    return mb_mwe, dhdt_mwe, fdiv_mwe


def exp_smooth_numba(da_data, da_thickness, n_thickn_scale, 
                     kernel_scale=None, min_ksize=2, max_ksize=None,
                     verbose=False):
    ''' Function to smooth input field
    Applies exponential filter, of variable size, to every pixel in the glacier.
    The filter size depends on local ice thickness.
    
    da_data         : field to smooth
    da_thickness    : local ice thickness that determines filtering length (decay of exponential weights)
    n_thickn_scale  : multiplication of local thickness. 
                      Rule of thumb: ice dynamics depend on 1-3 times the local ice thickness (valley glaciers) 
                      or up to 10 times ice thickness (ice sheets)
    kernel_scale    : option to scale the size of the filter kernel (px)  (don't use? its from another method)
    min_ksize       : minimum smoothing window (px)
    max_ksize       : cut-off of filtering (kernel for computational purposes). Can be None
        
        
    ## CODE SPEED UP OPTIONS: I tried options to lose the loop. 
    Challenge: varying kernel size for each pixel
    - use DASK arrays to parallize operations
        --> cannot do this as I need to be able to cross borders of chucnks for the variable kernel size
    - use xr.rolling() to apply moving window without explicit looping 
        --> cannot do this; requires fixed kernel size 
    - scipy.ndimage for fast convolutions, with custum kernel-based filtering 
        --> requires precomputing the kernel, too difficult withh varying ksize
    - numba JIT-compile loops: speed up of loops by compiling into fast machine code
        --> does not support dynamic memory allocation so only works with a double loop 
            (i.e. instead of matrix multiplication of the kernel, we need to loop 
             each kernel-pixel, adding another double-loop to thhe code)
    - cuPy for GPU acceleratino: can process many pixels simultaniously. 
        --> complex memory management (?)

    Conclusions:
    - Initial version with px-per-px loop in xarray: 4.2 min for aletsch glacier
    - Changing to use numpy array in px-per-px loop instead of xarray; 8.6 sec
    - Using numba nJit to speed up loop, requires to add another double-loop for every kernel: 3.6 sec / 2.5 sec depending on nan-handling

    Validation:
        The differences of exp_smooth_xarray vs exp_smooth_numpy are < 3.78e-07 ; floating point precision error
        The difference of exp_smooth_numpy vs exp_smooth_numba is significant... 
            --> found error with kernel weighths, including Nan px in total weight 
                of kernel for NUMPY and XARRAY calculations. Numba calculations are the correct ones :) 

    ## Available functions: from slow to fast
        - exp_smooth_xarray (ksize inside of loop; which is faster than outside of loop)
        - exp_smooth_numpy 
        - exp_smooth_numba
        
    '''
    
    resolution = np.abs(da_thickness.rio.resolution()[0])
    
    ''' ------
    ## Determine filter/smoothing length, i.e. the decay of the exponential , and resulting kernel size
    ----------'''
    
    ## exponential decay length [m] (smoothing length): scales with N * local ice thickness 
    da_exp_decay_length = n_thickn_scale * da_thickness # [m]
    
    ''' 
    ## kernel size [px]: define maximum distance, normalized by smoothinglength/res, 
    to be used for smoothing the data, as with using the full decaylength the code would become too slow. 
    - kernel_scale Should be at least 2 for decent results (Lander). 
   '''
    # da_ksize = (kernel_scale * da_exp_decay_length/resolution).round()
    da_ksize = (da_exp_decay_length/resolution).round()
    da_ksize = xr.where(da_ksize>min_ksize, da_ksize, min_ksize) # apply min-max sizing of kernel
    if max_ksize: # possible to have free max (then completely dpeendend on ice thickness) but this could make code slower (not in my experience so far, though)
        da_ksize = xr.where(da_ksize<max_ksize, da_ksize, max_ksize)
    
    
    ''' ------
    ## Convert xarray to numpy array for faster looping  and working with Numba
    ----------'''
    y_size, x_size = da_data.sizes['y'], da_data.sizes['x']
    
    ## get numpy arrays
    nd_data = da_data.values
    nd_thickness = da_thickness.values
    y_size, x_size = da_data.sizes['y'], da_data.sizes['x']
    
    nd_exp_decay_length = da_exp_decay_length.values  # [m]
    nd_ksize = da_ksize.values
    
    ''' to work with numba, dont use NaNs -- aha; issue is to not use np.nansum , but by using a double-double-loop that can be avoided easily'''
    
    # # Replace NaNs with -999 to avoid issues in numba
    # nd_data = np.where(np.isnan(nd_data), -999, nd_data)
    # nd_thickness = np.where(np.isnan(nd_thickness), -999, nd_thickness)
    nd_exp_decay_length = np.where(np.isnan(nd_exp_decay_length), 0, nd_exp_decay_length)
    
    start = time.time()
    
    ''' ------
    ## function to apply smoothing on every px
    ----------'''

    @njit(parallel=True)
    def apply_kernel_numba(nd_data, nd_thickness, nd_exp_decay_length, nd_ksize, nd_y, nd_x,  y_size, x_size ):
        result = np.full((y_size, x_size), np.nan)
        
        
        ''' Loop all pixels of glacier (dataArray) '''
        for yidx in prange(y_size):
            for xidx in prange(x_size): # Skip nodata pixels
                # if nd_data[yidx, xidx] == -999 or nd_thickness[yidx, xidx] == -999:
                if np.isnan(nd_data[yidx, xidx]) or np.isnan(nd_thickness[yidx, xidx]):
                    # print('skip')
                    continue
                
                ''' current pixel value to update with smoothing kernel '''
                px_x = nd_x[xidx]
                px_y = nd_y[yidx]
                
                ''' get filter decay length of current px, determines smoothing kernel size'''
                exp_decay_length = nd_exp_decay_length[yidx, xidx]
                # if exp_decay_length == -999:
                if np.isnan(exp_decay_length):
                    continue
                
                ksize = int(nd_ksize[yidx, xidx])
                
                ''' extract kernel around centerpx'''
                y_start, y_end = max(yidx - ksize, 0), min(yidx + ksize + 1, y_size)
                x_start, x_end = max(xidx - ksize, 0), min(xidx + ksize + 1, x_size)
                
                nd_kernel_data = nd_data[y_start:y_end, x_start:x_end]
                
                nd_kernel_x = nd_x[x_start:x_end]
                nd_kernel_y = nd_y[y_start:y_end]
                
                ''' initialize weigths of smoothing kernel '''
                sum_weights = 0.0
                weighted_value_xy = 0.0
                
                ''' loop all pxs in smoothing kernel, calculate weights '''
                for ky in range(len(nd_kernel_y)):
                    for kx in range(len(nd_kernel_x)):
                        # while centerpx is non-nan, kernel might cover nan-px so again check for data validity
                        if np.isnan(nd_kernel_data[ky,kx]):
                            continue
                        
                        ''' each px weight depends on distance to center-px, with exponential decay '''
                        # distance = np.sqrt((px_y - nd_kernel_y[ky]) ** 2 + (px_x - nd_kernel_x[kx]) ** 2)
                        distance = np.sqrt((nd_kernel_y[ky] - px_y) ** 2 + (nd_kernel_x[kx] - px_x) ** 2)
                        weight = np.exp( (-1 / exp_decay_length) * distance)
                        
                        ''' keep track of total weight, 
                        add weighted value of center-px (cumulative value) '''
                        sum_weights += weight
                        weighted_value_xy += nd_kernel_data[ky, kx] * weight
                        
                # print('sum', sum_weights)
                # print('wvalue', weighted_value_xy)
                if sum_weights == 0:
                    continue  # Avoid division by zero
                ''' normalize cumulative weighted center-px by dividing to total weights '''
                result[yidx, xidx] = weighted_value_xy / sum_weights
        
        return result
                
    ''' ------
    ## Apply smoothing
    ----------'''
    
    nd_smooth = apply_kernel_numba(nd_data, nd_thickness, 
                                       nd_exp_decay_length, nd_ksize, 
                                       da_data.y.values, da_data.x.values, y_size , x_size)
        
    
    '''# convert numpy array back to xarray'''
    da_smooth_numba = da_data.copy(data=nd_smooth)
    da_smooth_numba = da_smooth_numba.where(da_smooth_numba>-999,np.nan)

    
    end = time.time()
    dt = (end - start)/60
    if verbose:
        print('elapsed time numba (min): {:.1f} ({} sec)'.format(dt,dt*60))
    
    return da_smooth_numba 




def extrapolate_nan_2d(da,interp_method='linear'):
    """Fill NaNs in a 2D xarray.DataArray using nearest-neighbor extrapolation
    
    griddata(..., method='linear') does interpolation within known values.
    griddata(..., method='nearest') is used to extrapolate into NaN areas.
    filled[np.isnan(filled)] = extrapolated[...] merges the two for a full field.

    """    
    ## implementing NaN with chatgpt
    
    # Ensure 2D
    assert len(da.dims) == 2, "DataArray must be 2D"
    yname, xname = da.dims

    # Get coordinate grids
    X, Y = np.meshgrid(da[xname], da[yname])

    # Mask of valid data
    valid_mask = ~np.isnan(da.values)

    # Flatten coordinates and values for known data
    points = np.column_stack((X[valid_mask], Y[valid_mask]))
    values = da.values[valid_mask]

    # Target grid
    points_full = np.column_stack((X.ravel(), Y.ravel()))


    # Interpolate using 'linear': use 'nearest' for extrapolation beyond the convex hull
    filled = griddata(points, values, points_full, method=interp_method)
    extrapolated = griddata(points, values, points_full, method='nearest')  # fallback for extrapolation
    filled[np.isnan(filled)] = extrapolated[np.isnan(filled)]



    # Reshape back to original shape and return as xarray
    return da.copy(data=filled.reshape(da.shape))


def make_parameter_grid():
    """Create one grid containing all smoothing workflows.
    f000: no smoothing (N=0), only vary F parameter
    f001: smooth fluxdiv, for N [0, 10]
    f110: smooth gradients, for N [0, 10]
    f111: smooth gradients and fluxdiv: only for N [0,5]
    """
    fparam_values = np.arange(0.75, 1.01, 0.05)

    f111_grid = [
        (lscale_gradients, lscale_fdiv, fparam)
        for lscale_gradients, lscale_fdiv, fparam in itertools.product(
            np.arange(0, 6, 1),
            np.arange(0, 6, 1),
            fparam_values,
        )
    ]
    f001_f110_grid = [
        (lscale, lscale, fparam)
        for lscale, fparam in itertools.product(
            np.arange(6, 11, 1), fparam_values)
    ]

    return f111_grid + f001_f110_grid, None


def get_glacier_mb_bins_glamos(df_glamos_mb,glacier_name=None,glacier_id=None): # extract glamos dataset
    '''
    Retrieve values of GLAMOS dataset for selected glacier
    Either providing a glacier name (must be exact match) or glacier sgi-id

    Parameters
    ----------
    df_glamos_mb : Pandas Dataframe
        DataFrame of GLAMOS data.
    glacier_name : str
        Glacier name as defined in GLAMOS dataset. Must be exact correct str. 
        The default is None.
    glacier_id : str, optional
        Glacier sgi-id as defined in GLAMOS dataset. 
        If both glacier_name and glacier_id are provided, priority is given to glacier_id.
        The default is None.

    Raises
    ------
    ValueError
        If neither glacier_id or glacier_name are given.

    Returns
    -------
    df_binned_hmin: Pandas DataFrame
        DataFrame of selected glacier, subset of df_glamos_mb. Only columns with numerical values are returned.
    name,sgid: Tuple, containing two strings
        returning both glacier_name and glacier_id

    '''
    
    '''##--- select glacier---'''
    if glacier_id:
        # glacier_id = 'B36-26' # ID of Aletsch
        df_mb_glacier = df_glamos_mb.loc[df_glamos_mb['sgi-id']==glacier_id].copy()
        name = df_mb_glacier['name'].unique()[0]
        sgid=glacier_id
    elif glacier_name:
        df_mb_glacier = df_glamos_mb.loc[df_glamos_mb['name']==glacier_name].copy()
        sgid = df_mb_glacier['sgi-id'].unique()[0]
        name=glacier_name
    else:
        raise ValueError('Provide either ')
    
        
    ''' ---- ## Group data into elevation bins --------'''
    col_numvars = ['date_start','Bw','Bs','Ba','area','h_min','h_max']
    # col_str = ['name','sgi-id']
    
    df_binned_hmin = df_mb_glacier[col_numvars].groupby('h_min').agg(['min', 'max', 'mean', 'median'])

    return df_binned_hmin, (name, sgid)



def bin_da_to_elevbins(da, da_elev, binstep=100): # Bin 2D in elevation classes; get elevation bins from data
    '''
    Function to discretize dataArray into provided elevation intervals.
    The aggregation into the bin is by calcualting the mean of all pixels that fall within that bin.
    Other approaches can be implemented at a later stage.

    ## Area weight:
    pd.groupby_bins().mean() sums all values in each bin and divides by the pixel count. 
    Since different bins have different pixel counts (10 vs 200), they don't represent the same "information density."
    BUT since each pixel = 1 "unit area" → can use unweighted mean as it is the same practically
    Still, larger bins have higher statistical reliability, so also calculate SDs per bin to quantify spread of values within bin.
    
    Parameters
    ----------
    da : xr.DataArray
        Data to be discretized into elevation bins. Should have same shape and dimensions as da_elev

    da_elev : xr.DataArray
        Elevation values in [m].
        
    binstep : scalar
        Width of elevation bins. 
        Expected to 100m to evaluate against GLAMOS mass balance data in bins of 100m width.

    Returns
    -------
    bin_means_data : numpy ndarray (Nbins,) 
        Providing the mean datavalue of da per bin
    hmin_bin_edges : numpy ndarray (Nbins,) 
        Left-edges of each bin.

    '''
    
    '''## 1. Get elevation values from data ''' 
    
    # make sure arange includes last value by adding +binstep 
    elev_bins_hmin_avail = np.arange(int(np.floor(da_elev/100).min()*100), # 1567 is rounded to 1500 
                                int(np.floor(da_elev/100).max()*100 + binstep), # 4117 is rounded to 4100
                                step=binstep) # 1500 to o4000      
    
    
    
    ''' # 2. Extent elevation values to generate enough bins
    When providing [x1, x2, x3] bin edges, two groups are created, using x3 as the right-side of the last bin.
    The elev_bins_hmin_avail provides the range of left-side of the available elevation values
    If elev_bins_hmin_avail is [h1, h2, h3] and h3 represents the last left-edge.
    Extending to [h1, h2, h3, h4] will give last interval h3-h4
    '''
    # hmin_bin_edges = np.concat((elev_bins_hmin_avail,[elev_bins_hmin_avail[-1]+100, elev_bins_hmin_avail[-1]+200]))
    hmin_bin_edges = np.concat((elev_bins_hmin_avail,[elev_bins_hmin_avail[-1]+binstep]))

    ''' ## 3 apply binning and include stats '''
    binned_data = da.groupby_bins(           # 2D datavalue to bin
                            group = da_elev,    # 2D bin data to identify groups
                            bins  = hmin_bin_edges, # bin edges -- for edges [1, 2, 3] two bins will be returned 
                            right = True, #include rightmost edge. If True, then binis [1,2,3,5] indicate (1,2], (2,3], (3,4] ie. right-inclusive
                            include_lowest = True# True # whether the FRIST interval is left-inclusive, so      [1,2], (2,3], (3,4] 
                            )
    binned_px_counts = binned_data.count().values # counts number of valid pixels per bin (i.e. excluding NaN)
    bin_means_data = binned_data.mean().values # unweighted mean per bin
    bin_std = binned_data.std().values # std deviation
    bin_sem = bin_std / np.sqrt(binned_px_counts)  # standard error of mean
    
    return bin_means_data, hmin_bin_edges[:-1], bin_std, bin_sem  # return left-edges of bins only

def load_glamos_elevationbins(path):
    df_glamos_raw = pd.read_csv(path, header=[6,7,8], delimiter=';')

    '''## pre-processing of dataframe'''
    df_glamos_mb = df_glamos_raw.copy()
    df_glamos_mb.columns = df_glamos_mb.columns.droplevel([0, 2])  # Keep only the first level (header at line 6) and drop the others
    df_glamos_mb.rename(columns={'Unnamed: 0_level_1':'name', '(according to Swiss Glacier Inventory)':'sgi-id','Unnamed: 11_level_1':'observer'}, inplace=True)
    df_glamos_mb['name'] = df_glamos_mb['name'].astype(str)
    df_glamos_mb['sgi-id'] = df_glamos_mb['sgi-id'].astype(str)

    for date_col in ['date_start','date_end_winter','date_end']:
        df_glamos_mb[date_col] = pd.to_datetime(df_glamos_mb[date_col], format='%d/%m/%Y', errors='coerce').astype('datetime64[ns]')

    # columns
    df_varnames = pd.DataFrame(
                            data=np.array([
                                df_glamos_raw.columns.droplevel([1,2]), 
                                df_glamos_mb.columns, 
                                df_glamos_raw.columns.droplevel([0,1])
                            ]).T, 
                            columns=['name', 'variable', 'unit']
                        )
    dict_varnames = df_varnames.set_index('variable')[['unit', 'name']].to_dict(orient='index');
    unit = dict_varnames['Bs']['unit']
    unit = 'mm w.e.'

    '''## ['E22-16'] Vadret Pers in GLAMOS-MB is not included in GLAMOS SGI-shapefile. '''
    ## Instead, use Vadret da Morteratsch ['E22-03']
    # update sgi for morteratsch; keep name? 
    df_glamos_mb.loc[df_glamos_mb['sgi-id']=='E22-16','name'] = 'Vadret Pers and Vadret da Morteratsch'
    df_glamos_mb.loc[df_glamos_mb['sgi-id']=='E22-16','sgi-id'] = 'E22-03'

    return df_glamos_mb



def get_watershed_segmentation(da_data, segment_threshold=0, small_obj_threshold=40):
    """Segment a 2D xarray.DataArray into accumulation and ablation areas using watershed segmentation.
    The segmentation is based on a threshold value (MB=0), and small objects are removed to reduce noise.
    """
    from skimage import morphology, measure
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi

    def _watershed_labels(mask, small_obj_threshold):
        """Clean a binary mask and label its watershed-separated components."""

        ## remove small gaps / objects from the mask to avoid noise in the segmentation
        mask_clean = morphology.remove_small_objects( mask, max_size=small_obj_threshold )
        mask_clean = morphology.remove_small_holes( mask_clean, max_size=small_obj_threshold )
        
        # ''' Label connected components in the cleaned mask  '''
        markers = measure.label(mask_clean)
        # ''' Compute distance transform for watershed segmentation '''
        distance = ndi.distance_transform_edt(mask_clean)
        # ''' Apply watershed segmentation to separate connected regions '''
        labels = watershed(-distance, markers, mask=mask_clean)
        return labels

    
    np_data = da_data.values

    ## create watershed labels for accumulation and ablation areas
    labels_acc = _watershed_labels( np_data > segment_threshold, small_obj_threshold )
    labels_abl = _watershed_labels( np_data < segment_threshold, small_obj_threshold )

    '''------------
    ## combine the two raw masks into a single segmentation image
    ## use -1 values for ablation labels, +1 for accumulation labels

    Some more cleaning of the segmentation
    - convert ACCumulation patches to a merged polygon
    - identify which ABL segments are fully enclosed in the ACC polygon, and remove those from the ABL labels
    ---------------- '''
    mask_segmentation = -1 * labels_abl.astype(float) + 1 * labels_acc.astype(float)
    mask_segmentation[mask_segmentation == 0] = np.nan
    da_segmented = da_data.copy(data=mask_segmentation).rename('segment')

    ## to dataframe
    df_segmented = da_segmented.to_dataframe(name='segment').reset_index()
    df_segmented.dropna(subset=['segment'], inplace=True)
    gdf_segmented = gpd.GeoDataFrame(
        df_segmented,
        geometry=gpd.points_from_xy(df_segmented.x, df_segmented.y),
        crs=da_data.rio.crs,
    )

    ''' Get patch contour; Convert contour coordinates to spatial coordinates '''
    ## get contours of masks 
    contour_acc = measure.find_contours(labels_acc, level=0.5)
    points_acc = []
    for contour in contour_acc:
        for point in contour:
            row, col = point
            x, y = da_data.rio.transform() * (col, row)  # Convert pixel coordinates to spatial coordinates
            points_acc.append((x, y))
    points_acc = np.array(points_acc)

    # # print(f'Found {len(points_acc)} points in accumulation mask.')
    # if len(points_acc) == 0:
    #     print('WARNING: No points found in accumulation mask. Check the segment_threshold and small_obj_threshold parameters.')
    if len(points_acc) > 0:
        gdf_contours_acc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(points_acc[:, 0], points_acc[:, 1]),
            crs=da_data.rio.crs,
        )
        gdf_ela_convexhull_acc = gpd.GeoDataFrame(geometry=gdf_contours_acc.dissolve().convex_hull)
        buffer_distance = 0.02 * np.sqrt(da_data.shape[0] * da_data.shape[1]) * np.abs(da_data.rio.resolution()[0])
        gdf_ela_convexhull_acc = gdf_ela_convexhull_acc.buffer(buffer_distance)

        segments_to_remove = []
        for label, group in gdf_segmented.groupby('segment'):
            if label < 0: # check for ablation points if they are in the "accumulation area convex hull"
                points_in_polygon = group.within(gdf_ela_convexhull_acc.geometry.iloc[0])
                if points_in_polygon.all():
                    segments_to_remove.append(label)
    else:
        segments_to_remove = []

    da_segmented_clean = da_segmented.where(
        ~da_segmented.isin(segments_to_remove), 0.5
    )
    acc = da_segmented_clean.where(da_segmented_clean > 0, False).astype(bool)
    abl = da_segmented_clean.where(da_segmented_clean < 0, False).astype(bool)
    da_segmented_binary = (-1 * abl + 1 * acc).where(da_segmented_clean.notnull())
    return da_segmented_binary



def linregress_mb_elev(df):
    """Perform linear regression of MB vs elevation and return gradient and ELA.
    INPUT:
    ------
    df: Pandas DataFrame with columns 'elevation' (m) and 'MB' (m.w.e./yr)
    
    OUTPUT:
    -------
    gradient: slope of the linear regression (in m w.e. / km)
    ELA: equilibrium line altitude (in m)
    x_vals: x values for the regression line (for plotting)
    y_vals: y values for the regression line (for plotting)
    """

    x = df['elevation'].values
    y = df['MB'].values
    if len(x) < 2: # need at least 2 points for linear regression
        gradient = np.nan
        ELA = np.nan
        x_vals = None
        y_vals = None
    else:
        model = LinearRegression().fit(x.reshape(-1,1), y)
        gradient = model.coef_[0]
        intercept = model.intercept_
        ELA = -intercept/gradient

        x_vals = np.array([x.min(), x.max() ])
        y_vals = intercept + gradient * x_vals
    return gradient*1000, ELA, x_vals, y_vals



def get_binidx_to_start(smb_bin_means_f001, max_skip_bins=3, max_fraction=0.15):
    max_allowed_bins = int(np.ceil(len(smb_bin_means_f001) * max_fraction))
    if max_allowed_bins < max_skip_bins:
        max_skip_bins = max_allowed_bins
        # print(f'.. max_skip_bins reduced to {max_skip_bins} based on max_fraction={max_fraction:.2f} of available bins')

    downward_bins = np.flatnonzero(np.diff(smb_bin_means_f001) < 0)
    if downward_bins.size == 0 or downward_bins[0] > max_skip_bins:
        return 0

    downward_bins = downward_bins[downward_bins <= max_skip_bins]
    return min(downward_bins[-1] + 1, max_skip_bins)
    # return _find_binidx_to_start(smb_bin_means_f001, max_skip_bins)

def calculate_ELA_binned(smb_bin_means, hmin_binned):
    '''# Calculate ELA for a single SMB realization (binned to elevation) 
    - Identify idx of elevation-bin where SMB switches from negative to positive
    - Use local (bin) MB-gradient to calculate ELA
    '''
    smb_binned = smb_bin_means.copy()

    sign_changes = np.diff(np.sign(smb_binned))
    zero_crossings = np.where(sign_changes > 0)[0]

    h_ela = np.nan # initialize
    hbin_zero_crossings = np.nan # initialize
    # h_theoretic = np.nan
    # if zero_crossings.size <= 0: # no zero crossings, so ELA is outside of glacier elevation range.
            
    if (sign_changes == 0).all(): # all SMB values have same sign but are not NaN.
        if (smb_binned < 0).all():
            hbin_zero_crossings = hmin_binned[-1]

    elif zero_crossings.size > 0: 
        '''
        - if there are no zero crossings, all of the glacier is below ELA. Keep NaN value for ELA.
        - if there ARE zero-crossings, calculate the ELA within the bin that the crossing occurs, using the local (bin) MB-gradient
        - if there are multiple zero crossings: try with smoothing the binned MB values and re-evaluate
        '''
        while zero_crossings.size > 1:
            smb_binned_smooth = np.convolve(smb_binned, np.ones(3)/3, mode='same')
            new_zero_crossings = np.where(np.diff(np.sign(smb_binned_smooth)) > 0)[0]
            if new_zero_crossings.size > 0:
                zero_crossings = new_zero_crossings
                smb_binned = smb_binned_smooth  # update for next iteration
            else:
                warnings.warn('No zero crossings after smoothing; take last crossing')
                zero_crossings_laststep = np.where(np.diff(np.sign(smb_binned)) > 0)[0]
                zero_crossings = [zero_crossings_laststep[-1]]
                break

        if len(zero_crossings) != 1:
            raise RuntimeError('multiple or 0 zero-crossings after while-loop... which one to take?')

        idx = zero_crossings[0]
        hbin_zero_crossings = hmin_binned[idx] # store h_bin value where zero crossing occurs.

        # calculate local mb gradient (i.e. gradient within h_bin) and ELA value
        h0, smb0 = hmin_binned[idx], smb_binned[idx]
        h1, smb1 = hmin_binned[idx + 1], smb_binned[idx + 1]
        local_mb_grad = (smb1 - smb0) / (h1 - h0)
        if local_mb_grad > 0: # should be non-null and positive
            h_ela = h0 - smb0 / local_mb_grad 
            
            # verify smb is approx 0 at ELA
            verification_value = smb0 + (h_ela-h0) * local_mb_grad
            if not np.isclose(verification_value, 0, atol=1e-10):
                raise ValueError(f"Verification failed): SMB at ELA is {verification_value}, not approx 0.")
        else:
            raise ValueError(f"Non-positive local MB gradient: {local_mb_grad}")

    return h_ela, hbin_zero_crossings


def calculate_MB_gradient_glamos(smb_bin_means, hmin_bins, ela0, 
                            skip_first_bin=True, max_skip_bins=3, max_fraction=0.15):
    
    ## split data into accumulation and ablation bins, include bin with ELA estimate in both cases.
    if not np.isnan(ela0): 
        smb_binned_ablation = smb_bin_means[hmin_bins <= ela0] ## both use <= or >= to include bin in both cases
        smb_binned_accumul = smb_bin_means[hmin_bins >= ela0]
        hbin_ablation = hmin_bins[hmin_bins <= ela0]
        hbin_accumul = hmin_bins[hmin_bins >= ela0]
        # Flag if ELA is out of bounds
        if len(smb_binned_ablation) == len(smb_bin_means): # ablation is full range, so no accumulation bins
            ELA_oob = True
        else: # Ela within bound
            ELA_oob = False # ELA within bounds
    else: # if no initial ELA estimate was availale, its also OOB
        smb_binned_ablation = smb_bin_means ## use all bins if ELA is not within glacier elevation range
        hbin_ablation = hmin_bins
        smb_binned_accumul = None
        ELA_oob = True # flag out of bounds

    ## MB gradient in ablation area
    idx_start=0
    if len(smb_binned_ablation) > 3: # need at least a few bins for linreg
        if skip_first_bin:
            raise ValueError("Depracated; expecting this fucntion only to be used for GLAMOS data, where skip_first_bin is not used")
            ## define which bin to skip:  first bin:
            # idx_start = 1 if skip_first_bin else 0
            ## check max allowed number of bins based on max_fraction:
            max_allowed_bins = int(np.ceil(len(smb_bin_means) * max_fraction))
            if max_allowed_bins < max_skip_bins:
                max_skip_bins = max_allowed_bins
                print(f'.. max_skip_bins reduced to {max_skip_bins} based on max_fraction={max_fraction:.2f} of available bins')
            idx_start = _find_binidx_to_start(smb_binned_ablation, max_skip_bins)

        ## lin regression
        x_abl = hbin_ablation[idx_start:] ## skip first bin here
        y_abl = smb_binned_ablation[idx_start:]
        x_abl = x_abl[~np.isnan(y_abl)]
        y_abl = y_abl[~np.isnan(y_abl)]

        # # fit line to ablation area
        model_abl = LinearRegression().fit(x_abl.reshape(-1,1), y_abl)
        MB_gradient_ablation = model_abl.coef_[0]
        
        ## set up line (for plotting) 
        x_vals = np.array([hbin_ablation.min(), hbin_ablation.max() ]) #if xlim is None else np.array(xlim)
        y_vals = model_abl.intercept_ + model_abl.coef_[0] * x_vals
        
    else:
        MB_gradient_ablation = np.nan ## no
        x_vals = np.array([hbin_ablation.min(), hbin_ablation.max() ]) #if xlim is None else np.array(xlim)
        y_vals = None
        x_abl, y_abl = None, None
        
    ## MB gradient in accumulation area 
    if not ELA_oob:
        x_acc = hbin_accumul[~np.isnan(smb_binned_accumul)]
        y_acc = smb_binned_accumul[~np.isnan(smb_binned_accumul)]
        model_acc = LinearRegression().fit(x_acc.reshape(-1,1), y_acc)
        MB_gradient_accumul = model_acc.coef_[0]
    else:
        MB_gradient_accumul = None
    
    return MB_gradient_ablation, MB_gradient_accumul, ELA_oob, x_abl, y_abl,x_vals, y_vals

