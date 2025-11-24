import sys
import os
import traceback
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import t
import pandas_datareader as pdr
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas_market_calendars as mcal
import requests
import zipfile
import io

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

def _download_famafrench_csv_fallback(model_name: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Download Fama-French factors directly from CSV zip URLs as fallback.
    
    Parameters
    ----------
    model_name : str
        Model name ('F-F_Research_Data_Factors_daily', 'F-F_Research_Data_5_Factors_2x3_daily', 
        or 'F-F_Momentum_Factor_daily')
    start_date : str, optional
        Start date for filtering data (YYYY-MM-DD format)
    end_date : str, optional
        End date for filtering data (YYYY-MM-DD format)
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame with Fama-French factors, or None if download fails
    """
    # Map model names to CSV zip URLs
    csv_urls = {
        'F-F_Research_Data_Factors_daily': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip',
        'F-F_Research_Data_5_Factors_2x3_daily': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
        'F-F_Momentum_Factor_daily': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    }
    
    if model_name not in csv_urls:
        return None
    
    url = csv_urls[model_name]
    
    try:
        # Download the zip file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract CSV from zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Find the CSV file in the zip
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            if not csv_files:
                warnings.warn(f"No CSV file found in zip for {model_name}")
                return None
            
            # Read the first CSV file
            csv_content = zip_file.read(csv_files[0])
            
            # Parse CSV - Fama-French CSVs have a header row, then data
            # Skip description lines that start with non-numeric characters
            lines = csv_content.decode('utf-8').split('\n')
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line starts with a date (YYYYMMDD format) or is header
                if line[0].isdigit() or 'Mkt-RF' in line or 'SMB' in line or 'Mom' in line:
                    data_lines.append(line)
            
            if not data_lines:
                warnings.warn(f"No data lines found in CSV for {model_name}")
                return None
            
            # Parse into DataFrame
            from io import StringIO
            csv_string = '\n'.join(data_lines)
            df = pd.read_csv(StringIO(csv_string))
            
            # Fama-French CSVs typically have date as first column (YYYYMMDD format)
            # Convert date column to datetime
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
            df = df.set_index(date_col)
            
            # Filter by date range if specified
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            
            return df
            
    except Exception as e:
        warnings.warn(f"Error downloading {model_name} from CSV fallback URL: {e}")
        return None


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
            # Check if it's a 404 error or HTTP error
            error_str = str(e).lower()
            is_404 = '404' in error_str or 'not found' in error_str or ('http' in error_str and ('error' in error_str or 'exception' in error_str))
            
            if is_404:
                warnings.warn(f"404 error fetching {model} from pandas_datareader, trying CSV fallback URL...")
                try:
                    df = _download_famafrench_csv_fallback(model, start_date, end_date)
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        # Process the downloaded CSV data same as pandas_datareader result
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
                            mom_col = [col for col in df.columns if 'Mom' in col or 'mom' in col]
                            if len(mom_col) > 0:
                                cols = {mom_col[0]: 'mom'}
                            else:
                                cols = {'Mom': 'mom'}  # fallback
                        
                        # Only rename columns that exist
                        existing_cols = {k: v for k, v in cols.items() if k in df.columns}
                        if len(existing_cols) > 0:
                            df.rename(columns=existing_cols, inplace=True)
                        
                        # Remove duplicate mkt_rf from ff5 (already in ff3)
                        if models[model] == 'ff5' and 'mkt_rf' in df.columns:
                            df.drop(columns=['mkt_rf'], inplace=True)
                        
                        if odf is None:
                            odf = df
                        else:
                            odf = pd.concat([odf, df], axis=1)
                        warnings.warn(f"Successfully loaded {model} from CSV fallback URL")
                    else:
                        warnings.warn(f"CSV fallback also failed for {model}")
                except Exception as fallback_error:
                    warnings.warn(f"Error fetching {model} from CSV fallback: {fallback_error}")
            else:
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

def create_returns_parquet(
    tickers: List[str],
    output_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_source: str = 'yahoo',
    use_adjusted: bool = True
) -> pd.DataFrame:
    """
    Fetch stock data from pandas_datareader and create a returns parquet file
    with tickers as columns, trade dates as index, and daily log returns as values.
    
    This function:
    1. Fetches historical price data for each ticker from pandas_datareader
    2. Calculates daily log returns for each ticker
    3. Pivots the data to wide format (dates as index, tickers as columns)
    4. Saves the result to a parquet file
    
    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols to fetch (e.g., ['AAPL', 'GOOGL', 'MSFT'])
    output_path : str
        Path where the parquet file will be saved (e.g., 'data/market/quotes.parquet')
    start_date : str, optional
        Start date for fetching data (YYYY-MM-DD format), by default None
    end_date : str, optional
        End date for fetching data (YYYY-MM-DD format), by default None (today)
    data_source : str, optional
        Data source for pandas_datareader (e.g., 'yahoo', 'tiingo', 'stooq', 'iex'), 
        by default 'yahoo'
    use_adjusted : bool, optional
        If True, use adjusted close prices for calculating returns (accounts for splits/dividends),
        by default True
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with dates as index, tickers as columns, and log returns as values.
        Missing values are filled with 0.
    
    Examples
    --------
    >>> from eventstudies.util import create_returns_parquet
    >>> df = create_returns_parquet(
    ...     tickers=['AAPL', 'GOOGL', 'MSFT'],
    ...     output_path='data/market/returns.parquet',
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31'
    ... )
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for ticker in tickers:
        try:
            # Fetch data from pandas_datareader
            df = pdr.DataReader(
                ticker,
                data_source,
                start=start_date,
                end=end_date
            )
            
            if df.empty:
                warnings.warn(f"No data returned for {ticker}")
                continue
            
            # Select price column (adjusted close if available and requested, else close)
            if use_adjusted and 'Adj Close' in df.columns:
                price_col = 'Adj Close'
            elif 'Close' in df.columns:
                price_col = 'Close'
            elif 'close' in df.columns:
                price_col = 'close'
            else:
                warnings.warn(f"No close price column found for {ticker}. Available columns: {list(df.columns)}")
                continue
            
            # Extract price series
            prices = df[price_col].copy()
            
            # Filter to positive prices
            prices = prices[prices > 0].copy()
            
            # Get NYSE trading calendar
            nyse = mcal.get_calendar('NYSE')
            
            # Get valid trading days for the date range
            min_date = prices.index.min()
            max_date = prices.index.max()
            trading_days = nyse.valid_days(start_date=min_date, end_date=max_date)
            trading_days_set = set(pd.to_datetime(trading_days).date)
            
            # Filter prices to only trading days
            prices_df = pd.DataFrame({'date': prices.index, 'price': prices.values})
            prices_df['date_only'] = pd.to_datetime(prices_df['date']).dt.date
            prices_df = prices_df[prices_df['date_only'].isin(trading_days_set)].copy()
            prices_df = prices_df.drop(columns=['date_only'])
            prices_df = prices_df.set_index('date')
            prices = prices_df['price']
            
            if len(prices) == 0:
                warnings.warn(f"No trading day data for {ticker}")
                continue
            
            # Sort by date to ensure proper ordering
            prices = prices.sort_index()
            
            # Calculate log returns using previous trading day
            # Group by ticker and calculate log return using previous row (which is previous trading day)
            prev_prices = prices.shift(1)
            
            # Calculate log returns, handling division by zero
            log_returns = np.log(prices / prev_prices)
            
            # Create DataFrame with date, ticker, and log return
            returns_df = pd.DataFrame({
                'date': prices.index,
                'ticker': ticker,
                'logreturn': log_returns
            })
            
            # Filter out rows where we don't have a previous trading day
            returns_df = returns_df[returns_df['logreturn'].notna()].copy()
            
            if not returns_df.empty:
                all_data.append(returns_df)
            else:
                warnings.warn(f"No valid returns calculated for {ticker}")
                
        except Exception as e:
            warnings.warn(f"Error fetching data for {ticker}: {e}")
            import traceback
            warnings.warn(f"Traceback: {traceback.format_exc()}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully fetched for any ticker")
    
    # Combine all ticker data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert date to datetime if needed
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Normalize dates to date-only (remove time/timezone if present)
    if isinstance(combined_df['date'].dtype, pd.DatetimeTZDtype):
        combined_df['date'] = combined_df['date'].dt.tz_localize(None)
    combined_df['date'] = combined_df['date'].dt.normalize()  # Set time to 00:00:00
    
    # Pivot to wide format: dates as index, tickers as columns
    df_wide = combined_df.pivot(index='date', columns='ticker', values='logreturn')
    
    # Fill NaN with 0 (missing returns are treated as 0)
    df_wide = df_wide.fillna(0)
    
    # Rename index to 'date'
    df_wide.index.rename("date", inplace=True)
    
    # Sort by date
    df_wide = df_wide.sort_index()
    
    # Convert column names to string
    df_wide.columns = df_wide.columns.astype(str)
    
    # Save to parquet
    df_wide.to_parquet(output_path, index=True)
    
    print(f"Created returns parquet: {len(df_wide):,} dates × {len(df_wide.columns):,} tickers")
    print(f"Date range: {df_wide.index.min()} to {df_wide.index.max()}")
    print(f"Saved to: {output_path}")
    
    return df_wide
