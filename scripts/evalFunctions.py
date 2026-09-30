import matplotlib
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import xarray as xr


def read_glamos_dat_file(dat_file, n_header_lines=4):
    """Read a GLAMOS .dat stake measurement file into a pandas DataFrame.
    Args:
        file (str): Path to the .dat file.
    Returns:
        pd.DataFrame: DataFrame containing the stake measurement data.
        pd.DataFrame: DataFrame containing the units for each column.

    Files are structered with 4 header lines
    1. # glacier name info
    2. # column names (separated by ';')
    3. # units (separated by ';')
    4. # attribution line
    5+. data lines (separated by non-uniform whitespaces, so need to use delim_whitespace=True (depcrecated --> use sep='\\s+')
    """
    
    '''## load with pandas directly, paste in column headers'''
    try:
        pd_stake_data_raw = pd.read_csv(dat_file, sep=r"\s+", skiprows=n_header_lines, header=None, engine='python', encoding='utf-8')
        pd_stake_data_headerUnits = pd.read_csv(dat_file, sep='; ', skiprows=1, nrows=1, engine='python', encoding='utf-8')
    except UnicodeDecodeError:
        pd_stake_data_raw = pd.read_csv(dat_file, sep=r"\s+", skiprows=n_header_lines, header=None, engine='python',  encoding='latin-1')
        pd_stake_data_headerUnits = pd.read_csv(dat_file, sep='; ', skiprows=1, nrows=1, engine='python', encoding='latin-1')

    colnames = pd_stake_data_headerUnits.columns.tolist() 
    colnames[0] = 'measurement_name' # colnames[0].replace('# ','')  # clean first column name
    ## clean trailing whitespaces from colnames
    colnames = [name.strip() for name in colnames]
    ## update colnames
    pd_stake_data_raw.columns = colnames
    pd_stake_data_headerUnits.columns = colnames
    ## clean '#' from first entry in pd_stake_data_headerUnits
    pd_stake_data_headerUnits.iloc[0,0] = '(-)'
    pd_stake_data_headerUnits


    ## add date0 and date1 as datetime
    pd_stake_data_raw['date0_dt'] = pd.to_datetime(pd_stake_data_raw['date0'].astype(str), format='%Y%m%d')
    pd_stake_data_raw['date1_dt'] = pd.to_datetime(pd_stake_data_raw['date1'].astype(str), format='%Y%m%d')

    return pd_stake_data_raw, pd_stake_data_headerUnits
    # return pd_stake_data, pd_stake_data_headerUnits



def sample_mb_at_stake_locations(model_data_dict, stake_data,
                                       buffer_size_N=None, resolution_m=50):
    """Sample MB at stake location,and optionally buffer mean/std for each stake reading.

    Args:
        da_data_fXXX: Mapping of model labels (for example, ``'f000'``) to 2D
                        xarray DataArrays with ``x`` and ``y`` coordinates.
        stake_data: DataFrame containing ``x_pos`` and ``y_pos`` columns.
        buffer_size_N: Optional buffer half-width in pixels. ``None`` disables
            buffer statistics; a value of 1.5 on a 50 m grid selects a 3x3 window.
        resolution_m: Grid resolution in meters.

    Returns:
        A copy of ``stake_data`` with ``smb_<label>_point`` values and, when
        requested, ``smb_<label>_buffer_mean`` and ``smb_<label>_buffer_std``.
    """
    sampled_data = stake_data.copy()
    buffer_size = None if buffer_size_N is None else buffer_size_N * resolution_m

    for idx, row in sampled_data.iterrows():
        x_stake = row.x_pos
        y_stake = row.y_pos

        for fXXX_label, da_mb_fXXX in model_data_dict.items():
            if not ( # check if point is within the spatial extent
                da_mb_fXXX.x.min() <= x_stake <= da_mb_fXXX.x.max()
                and da_mb_fXXX.y.min() <= y_stake <= da_mb_fXXX.y.max()
            ):
                raise ValueError(
                    f'Stake at index {idx} with location ({x_stake}, {y_stake}) '
                    'is outside the dataArray extent.'
                )

            sampled_data.at[idx, f'mb_{fXXX_label}_point'] = da_mb_fXXX.interp(
                x=x_stake, y=y_stake, method='linear'
            ).item()

            if buffer_size is not None:
                x_coords = da_mb_fXXX['x'].values
                y_coords = da_mb_fXXX['y'].values
                x_slice = slice(x_stake - buffer_size, x_stake + buffer_size)
                if x_coords[0] > x_coords[-1]:
                    x_slice = slice(x_stake + buffer_size, x_stake - buffer_size)
                y_slice = slice(y_stake - buffer_size, y_stake + buffer_size)
                if y_coords[0] > y_coords[-1]:
                    y_slice = slice(y_stake + buffer_size, y_stake - buffer_size)

                buffer_da = da_mb_fXXX.sel(x=x_slice, y=y_slice)
                sampled_data.at[idx, f'mb_{fXXX_label}_buffer_mean'] = buffer_da.mean().item()
                sampled_data.at[idx, f'mb_{fXXX_label}_buffer_std'] = buffer_da.std().item()
                if fXXX_label == next(iter(model_data_dict)):
                    sampled_data.at[idx, 'mb_buffer_Npix'] = (
                        f'{buffer_da.sizes.get("y", 0)}x{buffer_da.sizes.get("x", 0)}'
                    )

    return sampled_data


# def get_smb_at_stake_locations(da_mb_fXXX, pd_stake_data, 
#                                fXXX_label='', buffer_size_N=3, resolution_m=50):
#     ''' get smb values at stake locations by averaging over a buffer area around the stake location
#     buffer_size: in meters, e.g., 25 m for 50 m resolution data
#     '''
#     buffer_size = buffer_size_N * resolution_m

#     for idx, row in pd_stake_data.iterrows():
#         x_stake = row.x_pos
#         y_stake = row.y_pos

#         # define buffer area
#         x_min = x_stake - buffer_size
#         x_max = x_stake + buffer_size
#         y_min = y_stake - buffer_size
#         y_max = y_stake + buffer_size

#         ## check if x_stake and y_stake are within the data extent
#         if not (da_mb_fXXX.x.min() <= x_stake <= da_mb_fXXX.x.max()) or not (da_mb_fXXX.y.min() <= y_stake <= da_mb_fXXX.y.max()):
#             raise ValueError(f'Stake at index {idx} with location ({x_stake}, {y_stake}) is outside the data extent. {da_mb_fXXX.x.min().item()}-{da_mb_fXXX.x.max().item()}, {da_mb_fXXX.y.min().item()}-{da_mb_fXXX.y.max().item()}')
        
#         # select data within buffer area
#         da_buffer_fXXX = da_mb_fXXX.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

#         ## put in dataframe
#         pd_stake_data.at[idx, f'mb_{fXXX_label}_buffer_mean'] = da_buffer_fXXX.mean().item()
#         pd_stake_data.at[idx, f'mb_{fXXX_label}_buffer_std'] = da_buffer_fXXX.std().item()
#         pd_stake_data.at[idx, 'mb_buffer_Npix'] = f'{da_buffer_fXXX.shape[0]}x{da_buffer_fXXX.shape[1]}'
    
#     return pd_stake_data


def load_stake_data(filepath, start_year, end_year=2020):
    """Load, convert, and period-filter preprocessed GLAMOS stake measurements."""
    stake_data = pd.read_csv(filepath)
    stake_data['mb_we'] = stake_data['mb_we'] / 1000.0
    stake_data['mb_error'] = stake_data['mb_error'] / 1000.0
    stake_data['date0_dt'] = pd.to_datetime(stake_data['date0'].astype(str), format='%Y%m%d')
    stake_data['date1_dt'] = pd.to_datetime(stake_data['date1'].astype(str), format='%Y%m%d')

    stake_data = stake_data.loc[
        (stake_data['date0_dt'].dt.year >= start_year)
        & (stake_data['date0_dt'].dt.year <= end_year)
    ].copy()
    years = stake_data['date0_dt'].dt.year
    return stake_data, years.min(), years.max()


def stake_data_get_weights_and_summary(stake_data):
    """Calculate temporally average stake MB info, including weights for each stake.
    Args:
        stake_data (pd.DataFrame): DataFrame containing stake measurements.
    Returns:
        stake_summary (pd.DataFrame): DataFrame containing summary statistics for each stake, 
            including mean, min, max, count, std of mb_we, mean mb_error, 
            first and last measurement dates, and stake weight.
        weighted_stake_data (pd.DataFrame): DataFrame containing the original stake data with the added stake weights, 
            which are the fraction of measurements for each stakeID relative to the total number of measurements.
        """

    ## temporal average of stake measurements for each stakeID; and other summary statistics (min, max, count, std)
    stake_summary = stake_data.groupby('matched_stakeID', observed=False).agg(
                        {'mb_we': ['mean', 'min', 'max', 'count', 'std']})
    stake_summary.columns = ['_'.join(column).strip() for column in stake_summary.columns]
    ## Calculate 5th and 95th percentiles of mb_we for each stakeID
    stake_summary_pct = stake_data.groupby('matched_stakeID', observed=False).agg(
        mb_we_pct05 = ('mb_we',lambda x: x.quantile(0.05)),
        mb_we_pct95 = ('mb_we',lambda x: x.quantile(0.95))
    )
    
    ## Additional summary statistics: mean mb_error, first and last measurement dates
    stake_summary_extra = stake_data.groupby('matched_stakeID', observed=False).agg(
        {'mb_error': 'mean', 'date0_dt': 'first', 'date1_dt': 'last'})

    ## assemble everything together
    stake_summary = pd.concat([stake_summary, stake_summary_pct, stake_summary_extra], axis=1).reset_index()
    
    ## Calculate weight as the fraction of measurements for each stakeID relative to the total number of measurements
    stake_summary['stake_weight'] = ( stake_summary['mb_we_count'] / stake_summary['mb_we_count'].sum() )
    ### Merge the stake_summary with the original stake_data to get the weights for each measurement
    weighted_stake_data = stake_data.merge(
                            stake_summary[['matched_stakeID', 'stake_weight']],
                            on='matched_stakeID',
                            how='left')
    return stake_summary, weighted_stake_data


def weighted_rmse(observations, predictions, weights):
    """Calculate the weighted Root Mean Square Error."""
    # normalize weights to sum to 1
    weights = weights / np.sum(weights)
    # Calculate the weighted RMSE
    rmse = np.sqrt(np.sum(weights * (observations - predictions) ** 2))
    return rmse

def calculate_weighted_rmse_at_stakes_all_fXXX(ds_glacier_mb, ds_glacier_mb_f111, 
                                      stake_summary, weighted_stake_data):
    """Calculate parameter-grid RMSE using weighted, temporally averaged stakes.
        Args:
            ds_glacier_mb (xr.Dataset)      : Dataset containing modeled MB for a glacier. For f000, f001, and f110 parameterizations.
            ds_glacier_mb_f111 (xr.Dataset) : Dataset containing modeled MB for a glacier with f111 parameterization.
            stake_summary : pd.DataFrame
                            DataFrame containing summary statistics for each stake, 
                            including mean, min, max, count, std of mb_we, mean mb_error, 
                            first and last measurement dates, and stake weight.
            weighted_stake_data :   pd.DataFrame
                            DataFrame containing the original stake data with the added stake weights per measurement.
        Returns:
            rmse_dataset (xr.Dataset): Dataset containing RMSE values for each parameterization.
                            xr.dataset (Nlength, Fparam, Ngrad, Nfdiv) 
                            with variables rmse_f000, rmse_f001, rmse_f110, rmse_f111
            pd_stake_summary (pd.DataFrame): 
                        DataFrame containing summary statistics for each stake, 
                        including mean, min, max, count, std of mb_we, mean mb_error, 
                        first and last measurement dates, and stake weight.
            """
    
    ''' ## extract my derived MB values at stake locations (interpolated from the model grid) and calculate RMSE '''
    predicted_by_stake = []
    predicted_by_stake_f111 = []
    
    for stake_id in stake_summary['matched_stakeID']:
        ## get all observations for current stake
        stake_rows = weighted_stake_data.loc[weighted_stake_data['matched_stakeID'] == stake_id]

        # extract MB at stake location pixel
        predictions = [
            ds_glacier_mb.interp(x=row.x_pos, y=row.y_pos, method='linear')
            for _, row in stake_rows.iterrows()
        ]
        predictions_f111 = [
            ds_glacier_mb_f111.interp(x=row.x_pos, y=row.y_pos, method='linear')
            for _, row in stake_rows.iterrows()
        ]

        # average if multiple stake readings exist for the same stakeID (e.g., multiple years of measurements)
        mean_prediction = xr.concat(predictions, dim='stakeReading').mean(dim='stakeReading')
        mean_prediction_f111 = xr.concat(
            predictions_f111, dim='stakeReading'
        ).mean(dim='stakeReading')

        # store in list
        predicted_by_stake.append(
            mean_prediction.expand_dims({'stakeID': [stake_id]}).drop_vars(
                ['x', 'y'], errors='ignore'
            )
        )
        predicted_by_stake_f111.append(
            mean_prediction_f111.expand_dims({'stakeID': [stake_id]}).drop_vars(
                ['x', 'y'], errors='ignore'
            )
        )

    ## assemble into xarray DataArrays
    predictions = xr.concat(predicted_by_stake, dim='stakeID', coords='all')
    predictions_f111 = xr.concat(predicted_by_stake_f111, dim='stakeID', coords='all')

    observed = xr.DataArray(
        stake_summary['mb_we_mean'].values,
        coords={'stakeID': stake_summary['matched_stakeID'].values},
        dims=['stakeID'],
    )
    weights = xr.DataArray(
        stake_summary['stake_weight'].values,
        coords={'stakeID': stake_summary['matched_stakeID'].values},
        dims=['stakeID'],
    )

    rmse = np.sqrt((weights * (predictions - observed) ** 2).sum(dim='stakeID', skipna=True))
    for variable in ('mb_f000', 'mb_f001', 'mb_f110'):
        rmse[variable] = rmse[variable].where(rmse[variable] != 0, np.nan)
    rmse_f111 = np.sqrt(
        (weights * (predictions_f111 - observed) ** 2).sum(dim='stakeID', skipna=True)
    )

    rmse_dataset = xr.Dataset(
        {
            'rmse_f000': (('Nlength', 'Fparam'), rmse['mb_f000'].data),
            'rmse_f001': (('Nlength', 'Fparam'), rmse['mb_f001'].data),
            'rmse_f110': (('Nlength', 'Fparam'), rmse['mb_f110'].data),
            'rmse_f111': (('Ngrad', 'Nfdiv', 'Fparam'), rmse_f111['mb_f111'].data),
        },
        coords={
            'Nlength': ds_glacier_mb['Nlength'].values,
            'Fparam': ds_glacier_mb['Fparam'].values,
        },
    )
    return rmse_dataset

 
def plot_smb_elevbins_with_stakes(df_mb_glacier, 
                                  smb_datasets, df_bestparams,
                                  smb_err_datasets, smb_err_type,
                                  hmin_binned,
                                  ):
    my_palette = ['#2b6f39','#efbb1a','#d490c6'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#efbb1a
    # my_palette4 = ['#181631','#2b6f39','#efbb1a','#d490c6'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#efbb1a
    my_palette4 = ['#2b6f39','#efbb1a','#d490c6','#2c7fb8'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#efbb1a
    
    '''-------------------
    get values for plotting
    ----------'''
    ## extract from dataset dictionaries
    smb_bin_means_f000 = smb_datasets['MB: no smoothing (f000)']
    smb_bin_means_f001 = smb_datasets['MB: smooth only f.div. (f001)']
    smb_bin_means_f110 = smb_datasets['MB: smooth only gradients (f110)']
    smb_bin_means_f111 = smb_datasets['MB: smooth both (f111)']

    smb_bin_err_f000 = smb_err_datasets['MB: no smoothing (f000)']
    smb_bin_err_f001 = smb_err_datasets['MB: smooth only f.div. (f001)']
    smb_bin_err_f110 = smb_err_datasets['MB: smooth only gradients (f110)']
    smb_bin_err_f111 = smb_err_datasets['MB: smooth both (f111)']


    if smb_err_type is not None:
        if smb_err_type == 'sem':
            ## calculate bin range of  Standard Error of Mean (as I have a value per F, N param, I want to aggregate to single uncertainty per bin. Use minmax range)
            smb_bin_err_f000 = np.array([np.nanmin(smb_bin_err_f000, axis=(0,1)), 
                                                np.nanmax(smb_bin_err_f000, axis=(0,1))]) # shape (2, h_bin)
            smb_bin_err_f001 = np.array([np.nanmin(smb_bin_err_f001, axis=(0,1)), 
                                                np.nanmax(smb_bin_err_f001, axis=(0,1))]) # shape (2, h_bin)
            smb_bin_err_f110 = np.array([np.nanmin(smb_bin_err_f110, axis=(0,1)), 
                                                np.nanmax(smb_bin_err_f110, axis=(0,1))]) # shape (2, h_bin)
            smb_bin_err_f111 = np.array([np.nanmin(smb_bin_err_f111, axis=(0,1)), 
                                                np.nanmax(smb_bin_err_f111, axis=(0,1))]) # shape (2, h_bin)
        if smb_err_type == 'std':
            ## for std makes more sense to generate a mean value of all std(F,N,hbin) values? otherwise not sure how to interpret

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) ## ignore 'mean of empty slice' warnings
                smb_bin_err_f000 = np.nanmean(smb_bin_err_f000, axis=(0,1)) 
                smb_bin_err_f001 = np.nanmean(smb_bin_err_f001, axis=(0,1))
                smb_bin_err_f110 = np.nanmean(smb_bin_err_f110, axis=(0,1))
                smb_bin_err_f111 = np.nanmean(smb_bin_err_f111, axis=(0,1))

    '''-------------------
    ## melt df to long-form
    -----------'''
    dfs_combined = []
    for name, data in smb_datasets.items():
        # Reshape (N, F, h_bin) to (N*F, h_bin) and convert to DataFrame
        # i.e.: flatten the N and F dimensions into a single dimension, then transpose so that each column represents one N,F combination
        reshaped = data.reshape(-1, data.shape[-1])
        df_plot = pd.DataFrame(reshaped.T, index=hmin_binned)
        
        # Plot all N*F combinations at once
        # ax.plot(df_plot.index, df_plot.values, alpha=0.7, label=f'SMB {name}')

        # Melt to long format for seaborn
        df_melted = df_plot.reset_index().melt(id_vars='index', var_name='param_combo', value_name='smb')
        df_melted['dataset'] = name
        df_melted.rename(columns={'index': 'elevation'}, inplace=True)
        dfs_combined.append(df_melted)

    df_all = pd.concat(dfs_combined, ignore_index=True)

    
    ''' --- ## Plot settings ---'''
    sns.set_theme()
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10,6))

    '''----------
    ## plot GLAMOS shaded
    -----------'''
    h_min_values = df_mb_glacier.index  # x-axis values
    ax.fill_between(h_min_values, df_mb_glacier[('Ba', 'min')]/1000, df_mb_glacier[('Ba', 'max')]/1000, color='gray', alpha=0.2, label='GLAMOS binned Min-Max')
    ax.plot(h_min_values, df_mb_glacier[('Ba', 'mean')]/1000, color='black', linestyle='-', label='GLAMOS binned Mean')
    

    '''----------
    ## my SMB data
    -----------'''

    '''-- simple lineplot with seaborn, shaded area and mean value as thick line ---'''

    # sns.lineplot(data=df_all, x='elevation', y='smb', 
    #             hue='dataset', palette=my_palette,
    #             estimator='mean', errorbar='sd', #errorbar=('pi',100), 
    #             linewidth=2,
    #             ax=ax) # pi=percentile interval, 100=all, so practically min-max range; sd = standard deviation
    # ylim=ax.get_ylim()


    '''-- lineplot with best-RMSE as thick line '''

    for df_fXXX, fXXX_colname, color , lstyle in zip(
                df_all['dataset'].unique(), 
                df_bestparams.columns.to_list(), 
                my_palette4, ['-.', '-', '-.', '--']):
        
        df_plot_allcombos  = df_all[df_all['dataset'] == df_fXXX]
        df_plot_bestparams = df_bestparams[fXXX_colname]
        
        # Plot shaded area (range of all parameter combinations :: or stdev of combinations? as sns.lineplot did. )
        min_smbs = df_plot_allcombos.groupby('elevation').min('smb').reset_index()
        max_smbs = df_plot_allcombos.groupby('elevation').max('smb').reset_index()
        ax.fill_between(min_smbs['elevation'], min_smbs['smb'], max_smbs['smb'], 
                    alpha=0.18, # 0.25, 
                    color=color, 
                    label=None, # f'{name} range'
                    )
        # ## plot each line individually with same color and low alpha to get the 'cloud' of lines effect, but without adding to legend
        # for param_combo, df_subset in df_plot_allcombos.groupby('param_combo'):
        #     ax.plot(df_subset['elevation'], df_subset['smb'], 
        #             color=color, alpha=0.1, linewidth=1, linestyle='-', label=None)  
        
        
        # Plot the thick line
        ax.plot(df_plot_bestparams.index, df_plot_bestparams, linewidth=2, 
                color=color, linestyle=lstyle,
                label=df_fXXX)
        
    ylim=ax.get_ylim()


    ''' --- add errorbars ---'''
    if smb_err_type is not None:
        if smb_err_type == 'sem':
            errlabel = 'standard error of mean'
            errjitter=False
        if smb_err_type == 'std':
            errlabel = 'standard-deviation of bin \n mean(stdev) across param space'
            errjitter=True
        ## add errorbar manually for SEM ranges: min-max range of SEM(N,F,hbin)
        hjitter = 0
        for smb_binmean, smb_err, ecol in zip([smb_bin_means_f000, smb_bin_means_f001, smb_bin_means_f110],
                                        [smb_bin_err_f000, smb_bin_err_f001, smb_bin_err_f110],
                                        my_palette):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) ## ignore 'mean of empty slice' warnings
                smb_tot_binmean = np.nanmean(smb_binmean, axis=(0,1))
            ax.errorbar(x=hmin_binned+hjitter, y=smb_tot_binmean, 
                        yerr=smb_err, 
                        fmt='none', ecolor=ecol, alpha=0.7, capsize=3,
                        label=errlabel)
            if errjitter:
                hjitter+=10  # small x-jitter to separate overlapping errorbars

    ax.hlines(0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color=[0.1,0.1,0.1], linestyle='--', alpha=0.5)
    ax.set_ylim(ylim)

    '''## get ax legend handles and labels'''
    # if pd_stake_data_yyyy is not None:
    #     handles, labels = ax.get_legend_handles_labels()
    #     ## replace maker-labels with year ranges
    #     idx_marker_labels = [i for i, lab in enumerate(labels) if lab in markerstyles.values()]
    #     labels_updated = labels.copy()
    #     for i, idx in enumerate(idx_marker_labels):
    #         labels_updated[idx] = stylemarkers[labels[idx]]  # each marker appears twice in legend (once for hue, once for style)
    #     ## set updated legend
    #     ax.legend(handles=handles, labels=labels_updated, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # else:
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # ax.set_title('Glacier SMB vs Elevation; shaded area = std dev; error bars = SEM range\n \
    #              NB: glamos selected 2010-2020; 2Dsmb is with hugonnet2015-20.')
    # ax.set_title(f'Glacier SMB vs Elevation; thickline = mean, shaded area = stdev; error bars = {smb_err_type} range\n' \
    #             f'Glamos stake range: {glamos_y0}-2020; ' \
    #             f'SMB with dHdt range: {dhdt_period}')

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return fig,ax



def plot_glacier_data(data, minmax,
                    #   glacier_sgiid, glacier_name,
                      rgi_outlines=None, sgi_outlines=None, 
                      cmap=matplotlib.cm.RdBu_r,
                      plot_scalebar=False ,clabel='',title='',ax=None,
                      add_colorbar=True, cbar_vertical = True,
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

    # ## set 'bad' values to yellow instead of transparent/gray 
    # if (cmap.get_bad() == 0).all():
    #     cmap.set_bad('#F7CF59',1.)
    if cbar_vertical: orientation = 'vertical'; 
    else: orientation = 'horizontal'
    ## plot
    if add_colorbar:
        data.plot.imshow(ax=ax,cmap=cmap,vmin=vmin, vmax=vmax, 
                        cbar_kwargs={'fraction':0.046,'pad':0.04,
                                    'label':clabel, 'orientation':orientation})  
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
    if rgi_outlines is not None:
        rgi_outlines.boundary.plot(ax=ax,color='k', label='RGI lines')
    
    # plt.tight_layout()
    return ax



def plot_stake_smb_scatter_ax(pd_stake_data, fXXX, #glacier_sgiid, glacier_name, #dhdt_period, 
                                xvarname='mb_we', errname = 'mb_error', msize = 40,
                                N_F_str = '', style_var=None, edgecolor='white',
                                hue_var='date0_yr', plot_palette=sns.color_palette("icefire" ,n_colors=24 )[:20],
                                rmse_value=None, add_legend=True,
                                ax=None
                           ):
    ''' plot scatter of stake MB vs modeled SMB at stake locations
    N_F_params_dict: dict with keys as smb types and values as (N,F) tuples
    '''
    if ax is None:
        fig, ax = plt.subplots(1,figsize=(8,6) )# sharey=True)
    else:
        fig = plt.gcf()

    
    ''' ---- MB FOR fXXX --- '''
    # ax=axs[0]
    sns.scatterplot(data=pd_stake_data, x=xvarname, y=f'mb_{fXXX}', 
                    hue=hue_var, palette=plot_palette, style=style_var,
                    # hue='measurement_name', palette='tab20',
                    edgecolor=edgecolor, linewidth=0.3,
                    s=msize, alpha=0.9,
                    ax=ax, legend=add_legend,
                    zorder=2.5) ## specify zorder for points to be above errorbars
    ## add VERTICAL errorbar using derived uncertainty around point, mb_fXXX_buffer_std
    ax.errorbar(pd_stake_data[xvarname], pd_stake_data[f'mb_{fXXX}'], 
                yerr=pd_stake_data[f'mb_{fXXX}_buffer_std'], 
                fmt='none', ecolor='gray', 
                alpha=0.7, capsize=None,
                barsabove=False,
                )
    ## add HORIZONTAL errorbar using observation mb_error for horizontal error
    ax.errorbar(pd_stake_data[xvarname], pd_stake_data[f'mb_{fXXX}'], 
                xerr=pd_stake_data[errname],
                fmt='none', ecolor='gray', 
                alpha=0.7, capsize=None,
                barsabove=False,
                )
    
    ## make sure mb=0 is at least included in axes
    xlim_min = ax.get_xlim()[0]; ylim_min = ax.get_ylim()[0]
    xlim_max = ax.get_xlim()[1]
    if xlim_max < 0:
        ax.set_xlim(xlim_min, 0.5)
        ax.set_ylim(xlim_min, 0.5)

    # get xlim for plotting 1:1 line
    xlim = ax.get_xlim()
    xlim = [xlim[0]-0.5, xlim[1]+0.5] ## make xlim a bit wider
    ax.plot(xlim, xlim, color='k', linestyle='--')  # 1:1 line
    ax.set_xlabel('GLAMOS Stake MB [m w.e./yr]')
    ax.set_ylabel(f'derived MB [m w.e./yr]')
    ax.set_title(f'MB-{fXXX} ({N_F_str})')
    # ax.set_title(f'smb-f000 (N={N_F_params_dict["f000"]["N"]}, F={N_F_params_dict["f000"]["F"]})')

    ## legend outside
    if add_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title=hue_var)

    ## annote RMSE value in a corner with a box around it, in the bottom-right corner
    if rmse_value:
        ax.text(0.95, 0.05, f'RMSE={rmse_value:.2f}', transform=ax.transAxes, fontsize=13,
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle='round', edgecolor='black', facecolor='white',alpha=0.8))
    # fig.suptitle(f'{glacier_sgiid}: {glacier_name}, {dhdt_period} ')
    return fig, ax


''' ##################################################
Function to plot N-F heatmaps
###################################################### '''
def plot_heatmap(heatmap_dataf001, heatmap_dataf110, xlabel, ylabel, 
                 clim=None, cmap=sns.cubehelix_palette(as_cmap=True), 
                 annotate_all=False,
                 axs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    ## get input
    if axs is None:
        fig,axs = plt.subplots(1,2, figsize=(16,6))#, sharey=True)
    else:
        fig = plt.gcf()
    if len(axs) != 2:
        raise ValueError('expected 2 axes for f001 and f110 heatmaps')
    
    if clim is None:
        clim = [np.nanmin([heatmap_dataf001.min(), heatmap_dataf110.min()]),
                np.nanmax([heatmap_dataf001.max(), heatmap_dataf110.max()])]

    '''### PLOT : heatmap F001'''

    ax=axs[0]
    sns.heatmap(heatmap_dataf001, xticklabels=xlabel, yticklabels=ylabel, cmap=cmap,
                ax=ax, vmin=clim[0], vmax=clim[1]
                )
    ax.set_title('Smooth only fluxdiv (F001)')
    ax.set_xlabel('F dimension')
    ax.set_ylabel('N dimension')

    # annotate best-performing N-F
    idx_nf = np.argmin(heatmap_dataf001)
    idx_f, idx_n = np.unravel_index(idx_nf, heatmap_dataf001.shape)
    
    ## annotate all best-performing N-F: can be multiple matches
    min_value_f001 = heatmap_dataf001.flatten()[idx_nf]
    indices_min = np.argwhere( heatmap_dataf001 == min_value_f001 ) # n,f
    if len(indices_min) > 1:
        for idx_f, idx_n in indices_min:
            ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))
    else:
        ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))      


    ## annotate all N-F values in very small font, an :.1f
    if annotate_all:
        for i in range(heatmap_dataf001.shape[0]):
            for j in range(heatmap_dataf001.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{heatmap_dataf001[i, j]:.2f}", color='black', ha='center', va='center', fontsize=6)
    # else:    # annotate only best-performing N-F
    #     ax.annotate(f"{heatmap_dataf001[idx_f,idx_n]:.2f}", xy=(idx_n + 0.7, idx_f + 0.5), color='black', ha='center', va='center')


    '''### PLOT : heatmap F110'''
    ax=axs[1]
    sns.heatmap(heatmap_dataf110, xticklabels=xlabel, yticklabels=ylabel, cmap=cmap,
                ax=ax, vmin=clim[0], vmax=clim[1]
                )
    ax.set_title('Smooth only gradients (F110)')
    ax.set_xlabel('F dimension')
    ax.set_ylabel('N dimension')

    # annotate best-performing N-F
    idx_nf = np.argmin(heatmap_dataf110)
    idx_f, idx_n = np.unravel_index(idx_nf, heatmap_dataf110.shape)
    ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))        
    
    ## annotate all best-performing N-F: can be multiple matches
    min_value_f110 = heatmap_dataf110.flatten()[idx_nf]
    indices_min = np.argwhere( heatmap_dataf110 == min_value_f110 ) # n,f
    if len(indices_min) > 1:
        for idx_f, idx_n in indices_min:
            ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))
    else:
        ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))      

    ## annotate all N-F values in very small font, an :.1f
    if annotate_all:
        for i in range(heatmap_dataf110.shape[0]):
            for j in range(heatmap_dataf110.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{heatmap_dataf110[i, j]:.2f}", color='black', ha='center', va='center', fontsize=6)
    # else:    # annotate only best-performing N-F
    #     ## only best one
    #     ax.annotate(f"{heatmap_dataf110[idx_f,idx_n]:.2f}", xy=(idx_n + 0.7, idx_f + 0.5), color='black', ha='center', va='center')

    
    # plt.show()
    return fig, axs, min_value_f001, min_value_f110


''' ##################################################
Function to plot N-F heatmaps
###################################################### '''
def plot_heatmap_ax(heatmap_data, ax=None, xlabel='', ylabel='', 
                 clim=None, cmap=sns.cubehelix_palette(as_cmap=True), 
                 annotate_all=False,
                 ):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    ## get input
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = plt.gcf()
    
    if clim is None:
        clim = [np.nanmin([heatmap_data.min()]), np.nanmax([heatmap_data.max()])]

    '''### PLOT : heatmap '''

    sns.heatmap(heatmap_data, xticklabels=xlabel, yticklabels=ylabel, cmap=cmap,
                ax=ax, vmin=clim[0], vmax=clim[1]
                )
    ax.set_title('Smooth only fluxdiv (F001)')
    ax.set_xlabel('F dimension')
    ax.set_ylabel('N dimension')

    # annotate best-performing N-F
    idx_nf = np.argmin(heatmap_data)
    idx_f, idx_n = np.unravel_index(idx_nf, heatmap_data.shape)
    
    ## annotate all best-performing N-F: can be multiple matches
    min_value = heatmap_data.flatten()[idx_nf]
    indices_min = np.argwhere( heatmap_data == min_value ) # n,f
    if len(indices_min) >= 1:
        for idx_f, idx_n in indices_min:
            ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))
    # else:
    #     ax.add_patch(Rectangle((idx_n, idx_f), 1, 1, fill=False, edgecolor='white', lw=1.5))      


    ## annotate all N-F values in very small font, an :.1f
    if annotate_all:
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{heatmap_data[i, j]:.1f}", color='black', ha='center', va='center', fontsize=6)
    # else:    # annotate only best-performing N-F
    #     ax.annotate(f"{heatmap_data[idx_f,idx_n]:.2f}", xy=(idx_n + 0.7, idx_f + 0.5), color='black', ha='center', va='center')

    # plt.show()
    return fig, ax, min_value

def plot_ela_split_regression_ax(ax_ela, df_ablation, df_accumulation, df_glacier_skip, 
                                             x_vals_regr, y_vals_regr, ELA, linestyle_ela = '-'):
    ''' Plot function to visualize how ELA is derived from the MB-elevation scatter, 
    and how the linear regression is fitted to the ablation/accumulation points.
    '''
    my_palette = ['#2b6f39','#efbb1a','#d490c6'] 
    ## scatter, values different colors
    sns.scatterplot(data=df_ablation, x='elevation', y='MB', ax=ax_ela,
                                color=my_palette[0], alpha=0.5, s=10)
    sns.scatterplot(data=df_accumulation, x='elevation', y='MB', ax=ax_ela,
                    color=my_palette[1], alpha=0.5, s=10)
    sns.scatterplot(data=df_glacier_skip, x='elevation', y='MB', ax=ax_ela,
                    color='red', alpha=0.5, s=10, label='excluded')
    ## mb gradient
    if x_vals_regr is not None and y_vals_regr is not None:
        ax_ela.plot(x_vals_regr, y_vals_regr, color=my_palette[0], linestyle='--',
                        alpha=1, linewidth=2, label='Linear regression')
    ## ELA line
    ax_ela.vlines(ELA, *ax_ela.get_ylim(), color=my_palette[0],
                    linestyle=linestyle_ela, alpha=0.5, label=f'ELA = {ELA:.2f} m')
    ## MB=0 line
    # ax_ela.hlines(0, *ax_ela.get_xlim(), color='black', linestyle=':', alpha=0.5, linewidth=0.5, label='')
    ax_ela.set_title('ELA split / regression')
    ax_ela.set_xlabel('Elevation [m a.s.l.]')
    ax_ela.set_ylabel('Mass balance [m w.e./yr]')
    ax_ela.legend(loc='best', fontsize='small')
    ax_ela.grid('on', alpha=0.2)
    return ax_ela

def plot_add_inset_mb_segmented(ax, da_segmented, inset_position=[0.60, 0.0, 0.36, 0.42], title='MB segmentation'):
    ''' Plot function to visualize how segmentation is used to derive ELA
    Used in combination with e.g. function plot_ela_split_regression_ax to add an inset showing the segmented MB map.
    '''
    ax_inset = ax.inset_axes(inset_position)
    h = da_segmented.plot.imshow(
        ax=ax_inset, cmap='RdBu', vmin=-2, vmax=2, add_colorbar=False
    )
    ax_inset.set_title(title, fontsize=10)
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    ax_inset.set_xlabel(''); ax_inset.set_ylabel('')
    ax_inset.set_aspect('equal')
    return ax_inset


def load_glamos_fixdate(filepath):
    """Load and normalize the GLAMOS fixdate table used for ELA/AAR summaries."""
    fixdate = pd.read_csv(filepath, header=[4, 5, 6], delimiter=';')
    fixdate.columns = fixdate.columns.droplevel([0, 2])
    fixdate.rename(
        columns={
            'Unnamed: 0_level_1': 'name',
            '(according to Swiss Glacier Inventory)': 'sgi-id',
            'Unnamed: 11_level_1': 'observer',
        },
        inplace=True,
    )
    for column in ('name', 'sgi-id'):
        fixdate[column] = fixdate[column].astype(str)
    for column in ('date_start', 'date_end_winter', 'date_end'):
        fixdate[column] = pd.to_datetime(
            fixdate[column], format='%Y-%m-%d', errors='coerce'
        )

    morteratsch = fixdate['sgi-id'] == 'E22-16'
    fixdate.loc[morteratsch, 'name'] = 'Vadret Pers and Vadret da Morteratsch'
    fixdate.loc[morteratsch, 'sgi-id'] = 'E22-03'
    return fixdate
