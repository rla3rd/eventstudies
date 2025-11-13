import sys
import os
import traceback
import numpy as np
import pandas as pd
from scipy.stats import t
import pandas_datareader as pdr
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

warnings.simplefilter(action='ignore', category=Warning)

def add_asterisks(pvalue):
    asterisks = ""
    if pvalue < 0.01:
        asterisks = "***"
    elif pvalue < 0.05:
        asterisks = "**"
    elif pvalue < 0.1:
        asterisks = "*"
    return asterisks


def plot(time, CAR, *, AR=None, CI=False, var=None, df=None, confidence=0.95):

    fig, ax = plt.subplots()
    ax.plot(time, CAR)
    ax.axvline(
        x=0, color="black", linewidth=0.5,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if CI:
        delta = np.sqrt(var) * t.ppf(confidence, df)
        upper = CAR + delta
        lower = CAR - delta
        ax.fill_between(time, lower, upper, color="black", alpha=0.1)

    if AR is not None:
        ax.vlines(time, ymin=0, ymax=AR)

    if ax.get_ylim()[0] * ax.get_ylim()[1] < 0:
        # if the y axis contains 0
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    return fig


def plot_dist(X, scale=1, seed=3836):
    sns.distplot(X)
    np.random.seed(seed)
    xn = np.random.normal(scale=0.25, size=1000) 
    fig = sns.distplot(xn, kde=True, hist=False)
    return fig


def to_table(columns, asterisks_dict=None, decimals=None, index_start=0):

    if decimals:
        if type(decimals) is int:
            decimals = [decimals] * len(columns)

        for key, decimal in zip(columns.keys(), decimals):
            if decimal:
                columns[key] = np.round(columns[key], decimal)

    if asterisks_dict:
        columns["Signif"] = map(
            add_asterisks, columns[asterisks_dict["pvalue"]]
        )

    df = pd.DataFrame.from_dict(columns)
    df.index += index_start
    return df

def get_date_idx(X, date: np.datetime64, n: int = 5):
    idx = None
    for i in range(n):
        index = np.where(X == date)[0]
        if len(index) > 0:
            idx =  index[0]
        else:
            date += np.timedelta64(1, "D")
    return idx

def get_logreturns(env=os.environ.get('ENV', 'development')):
    try:
        # read logreturns1d.parquet file
        data = pd.read_parquet(
            f"s3://veritydata-deltalake/{env}/quotes/logreturns1d.parquet",
            storage_options={"anon": False})
        data.reset_index(inplace=True)
    except Exception:
        msg = sys.exc_info()
        details = traceback.format_exc()
        msg = f"{msg}: {details}"
        print(msg)

    return data

def update_famafrench(start_date='2003-07-01', end_date=None):
    """
    Fetch and normalize Fama-French factors from pandas_datareader.
    
    This function fetches Fama-French 3-factor, 5-factor, and momentum data,
    normalizes the column names, and returns a DataFrame with normalized columns.
    
    Parameters
    ----------
    start_date : str, optional
        Start date for fetching Fama-French data (YYYY-MM-DD format), 
        by default '2003-07-01'
    end_date : str, optional
        End date for fetching Fama-French data (YYYY-MM-DD format), 
        by default None (fetches up to latest available)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized column names:
        - date: Date column
        - mkt_rf: Market risk premium
        - ff3_smb, ff3_hml, ff3_rf: Fama-French 3-factor columns
        - ff5_smb, ff5_hml, ff5_rmw, ff5_cma, ff5_rf: Fama-French 5-factor columns
        - mom: Momentum factor
        All factor values are in decimal format (percentages divided by 100)
    """
    models = {
        'F-F_Research_Data_Factors_daily': 'ff3',
        'F-F_Research_Data_5_Factors_2x3_daily': 'ff5',
        'F-F_Momentum_Factor_daily': 'momentum'
    }

    odf = None
    for model in models.keys():
        try:
            result = pdr.DataReader(model, 'famafrench', start=start_date, end=end_date)
            
            # pandas_datareader may return a dict with multiple DataFrames
            # Extract the DataFrame from the dict if needed
            if isinstance(result, dict):
                # Prioritize numeric keys (usually 0) which contain the DataFrame
                # Skip 'DESCR' and other string keys
                df = None
                
                # First, try numeric keys (0 is typically the DataFrame)
                # Sort keys separately by type to avoid mixed-type comparison errors
                numeric_keys = [k for k in result.keys() if isinstance(k, (int, float)) or (isinstance(k, str) and k.isdigit())]
                for key in sorted(numeric_keys, key=lambda x: (isinstance(x, str), x)):
                    value = result[key]
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break
                
                # If not found in numeric keys, try all values
                if df is None:
                    for key, value in result.items():
                        # Skip string keys like 'DESCR'
                        if isinstance(key, str) and not key.isdigit():
                            continue
                        if isinstance(value, pd.DataFrame):
                            df = value
                            break
                
                # Verify we have a DataFrame
                if df is None:
                    warnings.warn(f"Could not extract DataFrame from {model} result (dict with keys: {list(result.keys())}, types: {[type(v) for v in result.values()]})")
                    continue
                
                if not isinstance(df, pd.DataFrame):
                    warnings.warn(f"Extracted value is not a DataFrame for {model}: {type(df)}")
                    continue
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                warnings.warn(f"Unexpected return type for {model}: {type(result)}")
                continue
            
            # Final verification: ensure df is a DataFrame before proceeding
            if not isinstance(df, pd.DataFrame):
                warnings.warn(f"df is not a DataFrame after extraction for {model}: {type(df)}")
                continue
            
            if models[model] == 'ff3':
                cols = {
                    'Mkt-RF': 'mkt_rf',
                    'SMB': 'ff3_smb',
                    'HML': 'ff3_hml',
                    'RF': 'ff3_rf'
                }
            elif models[model] == 'ff5':
                cols = {
                    'Mkt-RF': 'mkt_rf',
                    'SMB': 'ff5_smb',
                    'HML': 'ff5_hml',
                    'RMW': 'ff5_rmw',
                    'CMA': 'ff5_cma',
                    'RF': 'ff5_rf'
                }
            else:  # momentum
                # Handle both 'Mom' and 'Mom   ' (with spaces) column names
                # Verify df is a DataFrame before accessing columns
                if not isinstance(df, pd.DataFrame):
                    warnings.warn(f"df is not a DataFrame before momentum column processing: {type(df)}")
                    continue
                mom_col = [col for col in df.columns if 'Mom' in col or 'mom' in col]
                if len(mom_col) > 0:
                    cols = {mom_col[0]: 'mom'}
                else:
                    cols = {'Mom': 'mom'}  # fallback
            
            # Verify df is still a DataFrame before renaming
            if not isinstance(df, pd.DataFrame):
                warnings.warn(f"df is not a DataFrame before rename: {type(df)}")
                continue
            
            # Only rename columns that exist in the DataFrame
            existing_cols = {k: v for k, v in cols.items() if k in df.columns}
            if len(existing_cols) > 0:
                df.rename(columns=existing_cols, inplace=True)
            else:
                warnings.warn(f"No matching columns found for {model}. Expected: {list(cols.keys())}, Found: {list(df.columns)}")
                continue
            
            # Remove duplicate mkt_rf from ff5 (already in ff3)
            if models[model] == 'ff5' and 'mkt_rf' in df.columns:
                df.drop(columns=['mkt_rf'], inplace=True)
            
            if odf is None:
                odf = df
            else: 
                odf = pd.concat([odf, df], axis=1)
        except Exception as e:
            warnings.warn(f"Error fetching {model}: {e}")
            import traceback
            warnings.warn(f"Traceback: {traceback.format_exc()}")
    
    if odf is None:
        raise ValueError("Failed to fetch Fama-French data from pandas_datareader")
    
    # Convert index to date column and ensure proper format
    odf = odf.reset_index()
    odf.rename(columns={odf.columns[0]: 'date'}, inplace=True)
    odf['date'] = pd.to_datetime(odf['date'])
    
    # Convert percentage values to decimals (Fama-French data is in percentages)
    for key in odf.columns:
        if key != "date":
            odf[key] = odf[key] / 100
    
    return odf
