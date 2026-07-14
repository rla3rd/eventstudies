# Copyright (C) 2023 Richard Albright
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Tiingo data downloader and log returns preprocessor for eventstudies.
"""

import os
import time
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests

logger = logging.getLogger(__name__)


def download_prices(
    tickers: List[str],
    api_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    delay: float = 0.5
) -> pd.DataFrame:
    """
    Download historical daily price data from Tiingo API for a list of tickers.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to download.
    api_key : str, optional
        Tiingo API token. If None, it will attempt to fetch it from the
        TIINGO_API_KEY environment variable.
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.
    delay : float, optional
        Spaced delay in seconds between requests to honor Tiingo rate limits,
        by default 0.5.

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with columns: ['date', 'ticker', 'adjClose'].

    Raises
    ------
    ValueError
        If no API key is provided and the environment variable is not set.
    """
    token = api_key or os.getenv("TIINGO_API_KEY")
    if not token:
        raise ValueError(
            "Tiingo API Key is required. Please pass it as `api_key` or "
            "set the `TIINGO_API_KEY` environment variable."
        )

    base_url = "https://api.tiingo.com/tiingo/daily"
    headers = {"Authorization": f"Token {token}"}
    all_data = []

    session = requests.Session()

    for idx, ticker in enumerate(tickers):
        ticker = ticker.upper().strip()
        url = f"{base_url}/{ticker}/prices"
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        logger.info(f"Downloading {ticker} ({idx + 1}/{len(tickers)})...")
        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data:
                logger.warning(f"No data returned for {ticker}")
                continue

            df = pd.DataFrame(data)
            df["ticker"] = ticker
            # Keep only the columns we need for log returns
            if "date" in df.columns and "adjClose" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                all_data.append(df[["date", "ticker", "adjClose"]])
            else:
                logger.warning(f"Required columns missing in API response for {ticker}")
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

        # Spacing to prevent rate-limit throttling
        if len(tickers) > 1 and idx < len(tickers) - 1:
            time.sleep(delay)

    if not all_data:
        return pd.DataFrame(columns=["date", "ticker", "adjClose"])

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def to_logreturns(
    prices_df: pd.DataFrame,
    calendar: str = "NYSE"
) -> pd.DataFrame:
    """
    Transform long-format price data into wide-format log returns.

    Filters dates to the trading days of the specified calendar, computes
    consecutive log returns (using t-1 trading day), pivots, and fills missing values.

    Parameters
    ----------
    prices_df : pandas.DataFrame
        DataFrame containing columns: ['date', 'ticker', 'adjClose'].
    calendar : str, optional
        Name of the trading calendar (e.g., 'NYSE', 'NASDAQ', 'LSE'),
        by default 'NYSE'. Refer to pandas_market_calendars documentation.

    Returns
    -------
    pandas.DataFrame
        Wide-format log returns DataFrame with date as index and tickers as columns.
    """
    if prices_df.empty:
        return pd.DataFrame()

    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["adjClose"] > 0].copy()

    # Filter to trading days of the given calendar
    logger.info(f"Filtering to valid trading days for calendar: {calendar}")
    cal = mcal.get_calendar(calendar)
    min_date = df["date"].min()
    max_date = df["date"].max()
    trading_days = cal.valid_days(start_date=min_date, end_date=max_date)
    trading_days_set = set(trading_days.date)

    df["date_only"] = pd.DatetimeIndex(df["date"]).date
    df = df[df["date_only"].isin(trading_days_set)].copy()
    df = df.drop(columns=["date_only"])

    if df.empty:
        logger.warning("No data remains after calendar filtering.")
        return pd.DataFrame()

    # Calculate log returns using previous trading day (t-1)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["prev_adjClose"] = df.groupby("ticker")["adjClose"].shift(1)
    df["logreturn"] = np.log(df["adjClose"] / df["prev_adjClose"])

    # Drop rows without previous trading day
    df = df.dropna(subset=["logreturn"]).copy()

    # Ensure index date is normalized
    if isinstance(df["date"].dtype, pd.DatetimeTZDtype):
        df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()

    # Pivot to wide format
    df_wide = df.pivot(index="date", columns="ticker", values="logreturn")
    df_wide = df_wide.fillna(0.0)
    df_wide.index.rename("date", inplace=True)
    df_wide = df_wide.sort_index()
    df_wide.columns = df_wide.columns.astype(str)

    return df_wide
