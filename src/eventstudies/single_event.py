import os
import numpy as np
import pandas as pd
from scipy.stats import t
import s3fs
import warnings
import pandas_datareader as pdr


from .util import (
    to_table, 
    plot, 
    get_date_idx,
    get_logreturns,
    update_famafrench)


from .exception import (
    ParameterMissingError,
    DateMissingError,
    DataMissingError,
    ColumnMissingError,
    ReturnsCacheEmptyError,
    EventFormatError,
    EventKeyError)

from .models import (
    OrdinaryReturnsModel,
    MarketAdjustedModel,
    MeanAdjustedModel,
    MarketModel,
    FamaFrench3,
    FamaFrench5,
    Carhart)

warnings.simplefilter(action='ignore', category=Warning)


class SingleEvent:
    """
    Main event studies object.
    """

    _parameters = {
        "max_iteration": 5,
    }

    def __init__(
        self,
        model_func,
        model_data: dict,
        ticker: str = None,
        mkt_idx: str = None,
        event_date: np.datetime64 = None,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        description: str = None
    ):
        """
        Parameters
        ----------
        model_func : str
        model_data : dict
            Dictionary containing all parameters needed by `model_func`.
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
        est_size : int, optional
            Size of the estimation window for returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object

        Example
        -------
        Run an event study based on :
        .. the `MarketModel` function defined in the `models` submodule,
        .. given values for security and market returns,
        .. and default parameters
        >>> from eventstudies import SingleEvent, models
        >>> event = SingleEvent(
        ...     models.MarketModel, 
        ...     {'daily_ret':[0.032,-0.043,...], 'daily_mkt':[0.012,-0.04,...]}
        ... )
        """
        self.ticker = ticker
        self.mkt_idx = mkt_idx
        self.event_date = event_date
        self.event_window = event_window
        self.event_weight = event_weight
        self.win_size = -event_window[0] + event_window[1] + 1
        self.est_size = est_size
        self.buffer_size = buffer_size
        self.weight = weight
        self.description = description

        model = model_func(
            **model_data,
            est_size=self.est_size,
            win_size=self.win_size)

        self.AR, self.df, self.var_AR, self.model = model
        self.__compute()

    def __compute(self):
        self.CAR = np.cumsum(self.AR)
        self.var_CAR = [(i * var) for i, var in enumerate(self.var_AR, 1)]
        self.tstat = self.CAR / np.sqrt(self.var_CAR)
        # see https://www.statology.org/t-distribution-python/
        self.pvalue = (1.0 - t.cdf(abs(self.tstat), self.df)) * 2

    def results(self, asterisks: bool=True, decimals=4):
        """
        Return event study's results in a table format.
        
        Parameters
        ----------
        asterisks : bool, optional
            Add asterisks to CAR value based on significance of p-value, by default True
        decimals : int or list, optional
            Round the value with the number of decimal specified, by default 4.
            `decimals` can either be an integer, or a list of length 6.
        
        Note
        ----
        When `asterisks` is set as True, A significance (Signif) column ia added, 
        according to the CARs P-values.

        Returns
        -------
        pandas.DataFrame
            AR and AR's standard deviation, CAR and CAR's standard deviation, T-stat and P-value, 
            for each T in the event window.
        Note
        ----
        
        The function return a fully working pandas DataFrame.
        All pandas method can be used on it, especially exporting method (to_csv, to_excel,...)
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the market model with the S&P500 index as a market proxy.

        >>> event = SingleEvent.MarketModel(
        ...     ticker='AAPL',
        ...     mkt_idx='SPY',
        ...     event_date=np.datetime64('2007-01-09'),
        ...     event_window=(-5,+5)
        ... )
        >>> event.results(decimals = [3,5,3,5,2,2])

        Note
        ----
        
        Significance (Signif) level: r'***' at 99%, r'**' at 95%, r'*' at 90%
        """

        columns = {
            "AR": self.AR,
            "StdErr AR": np.sqrt(self.var_AR),
            "CAR": self.CAR,
            "StdErr CAR": np.sqrt(self.var_CAR),
            "T-stat": self.tstat,
            "P-value": self.pvalue,
        }

        asterisks_dict = {"pvalue": "P-value"} if asterisks else None

        return to_table(
            columns,
            asterisks_dict=asterisks_dict,
            decimals=decimals,
            index_start=self.event_window[0],
        )

    def plot(self, *, AR=False, CI=True, confidence=0.95):
        """
        Plot the event study result.
        
        Parameters
        ----------
        AR : bool, optional
            Add to the figure a bar plot of AR, by default False
        CI : bool, optional
            Display the confidence interval, by default True
        confidence : float, optional
            Set the confidence level, by default 0.95
        
        Returns
        -------
        matplotlib.figure
            Plot of CAR and AR (if specified).
        Note
        ----
        The function return a fully working matplotlib function.
        You can extend the figure and apply new set-up with matplolib's method (e.g. savefig).
        
        Example
        -------
        Plot CAR (in blue) and AR (in black), with a confidence interval of 95% (in grey).
        >>> event = SingleEvent.MarketModel(
        ...     ticker = 'AAPL',
        ...     mkt_idx = 'SPY',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        >>> event.plot(AR = True, confidence = .95)
        .. image:: /_static/single_event_plot.png
        """

        return plot(
            time=range(self.event_window[0], self.event_window[1] + 1),
            CAR=self.CAR,
            AR=self.AR if AR else None,
            CI=CI,
            var=self.var_CAR,
            df=self.df,
            confidence=confidence,
        )

    @classmethod
    def _save_parameter(cls, param_name: str, data):
        cls._parameters[param_name] = data

    @classmethod
    def _get_parameters(
        cls,
        param_name: str,
        columns: tuple,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1
    ) -> tuple:

        # Find index of returns
        try:
            event_i = get_date_idx(
                cls._parameters[param_name]["date"],
                event_date,
                cls._parameters["max_iteration"],
            )
        except KeyError:
            raise ParameterMissingError(param_name)

        if event_i is None:
            raise DateMissingError(event_date, param_name)

        start = event_i - (-event_window[0] + buffer_size + est_size)
        end = event_i + event_window[1] + 1
        size = -event_window[0] + buffer_size + est_size + event_window[1] + 1

        results = list()
        for column in columns:
            try:
                result = cls._parameters[param_name][str(column)][start:end]
            except KeyError:
                raise ColumnMissingError(param_name, column)

            # test if all data has been retrieved
            if len(result) != size:
                msg = ", ".join([
                    f"event_date: {event_date}",
                    f"event_idx: {event_i}",
                    f"result: {len(result)}", 
                    f"size: {size}",
                    f"start: {start}",
                    f"end: {end}",
                    f"weight: {weight}"])
                raise DataMissingError(param_name, column, len(result), start + end, msg)

            results.append(result)

        return tuple(results)

    @classmethod
    def import_returns(
        cls,
        *,
        path=None,
        env=os.environ.get('ENV', 'development'),
        dataframe=None
    ):
        """
        Import returns from a parquet file or pandas DataFrame for the `SingleEvent` Class parameters.
        Data should be in wide format with:
        - 'date' column (or date index) containing dates
        - Ticker symbols as column names (e.g., 'AAPL', 'SPY', etc.)
        - Each cell contains log returns for that ticker on that date
        
        Once imported, the returns are shared among all `SingleEvent` instances.
        
        The data format is compatible with:
        - Tiingo data (tiingo-logreturns.parquet format)
        - pandas_datareader output (after pivoting to wide format)
        - Any DataFrame with dates as index/column and tickers as columns

        
        Parameters
        ----------
        path : str, optional
            Path to the returns parquet file. The file should have:
            - A 'date' column (or date index)
            - Ticker symbols as column names
            - Log returns as values
        env : str, optional
            Environment for S3 path (if path is None), by default 'development'
        dataframe : pandas.DataFrame, optional
            Directly provide a DataFrame instead of reading from file.
            DataFrame should have dates as index and tickers as columns.
            If provided, 'path' is ignored.
        """
        if dataframe is not None:
            data = dataframe.copy()
            # Ensure date is a column, not index
            if data.index.name == 'date' or isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
            # Ensure date column exists and is datetime
            if 'date' not in data.columns:
                raise ValueError("DataFrame must have a 'date' column or date index")
            data['date'] = pd.to_datetime(data['date'])
        elif path is None:
            lr_base_uri = f"veritydata-deltalake/{env}/quotes/logreturns1d.parquet"
            fs = s3fs.S3FileSystem(anon=False)
            lr_uri = f"s3://{lr_base_uri}"
            try:
                files = fs.glob(lr_base_uri)
                if len(files) > 0:
                    # load the file if it exists
                    data = pd.read_parquet(lr_uri, storage_options={"anon": False})
                    data.reset_index(inplace=True)
                    print(f"Cached Returns: {data.shape}")
                else:
                    raise ReturnsCacheEmptyError
            except ReturnsCacheEmptyError:
                print("Cache Empty")
                return
        else:
            try:
                data = pd.read_parquet(path)
                # If date is in index, reset it
                if data.index.name == 'date' or isinstance(data.index, pd.DatetimeIndex):
                    data = data.reset_index()
                # Ensure date column exists
                if 'date' not in data.columns:
                    raise ValueError(f"File at {path} must have a 'date' column or date index")
                data['date'] = pd.to_datetime(data['date'])
            except Exception as e:
                print(f"Error with path: {e}")
                return
        
        # Ensure all ticker columns are strings (for compatibility)
        ticker_cols = [col for col in data.columns if col != 'date']
        data.columns = ['date'] + [str(col) for col in ticker_cols]
        
        data.fillna(0, inplace=True)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        cls._save_parameter("returns", data)

    @classmethod
    def import_FamaFrench(cls, start_date='2003-07-01', end_date=None):
        """
        Import Fama-French factors from pandas_datareader to the `SingleEvent` Class parameters.
        Once imported, the factors are shared among all `SingleEvent` instances.
        
        This method uses the normalized column names from update_famafrench():
        - mkt_rf: Market risk premium
        - ff3_smb, ff3_hml, ff3_rf: Fama-French 3-factor columns
        - ff5_smb, ff5_hml, ff5_rmw, ff5_cma, ff5_rf: Fama-French 5-factor columns
        - mom: Momentum factor
        
        Parameters
        ----------
        start_date : str, optional
            Start date for fetching Fama-French data (YYYY-MM-DD format), 
            by default '2003-07-01'
        end_date : str, optional
            End date for fetching Fama-French data (YYYY-MM-DD format), 
            by default None (fetches up to latest available)
        """
        data = update_famafrench(start_date=start_date, end_date=end_date)
        cls._save_parameter("FamaFrench", data)

    @classmethod
    def filter_event_returns(
        cls,
        df,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 30,
        famafrench: bool = False
    ):
        """
        convenience method to filter events using the returns 
        table so it doesn't have to be manually figured out
        df can be a pandas dataframe or a list of dict
        the method returns a tuple of pandas dataframes
        a filtered dataframe of events, and an excluded dataframe
        of events
        """
        colkeys = ('ticker', 'event_date')
        if type(df) == list:
            if len(df) == 0:
                return df
            else:
                for d in df:
                    if type(d) != dict:
                        raise EventFormatError
                    else:
                        if not all(k in d for k in colkeys):
                            for k in colkeys:
                                if k not in d:
                                    raise EventKeyError(k)
            df = pd.DataFrame(df)
        else:
            if not all(k in df.columns for k in colkeys):
                for k in colkeys:
                    if k not in df.columns:
                        raise EventKeyError(k)
            
        if famafrench:
            min_ff = cls._parameters['FamaFrench']['date'].min()
            max_ff = cls._parameters['FamaFrench']['date'].max()
            df = df[
                (df['event_date'] >= min_ff) 
                & (df['event_date'] <= max_ff)]
            
        start_retidx = np.max([est_size + buffer_size - event_window[0], 0])
        end_retidx = event_window[1]
        min_retdate = cls._parameters['returns']['date'].iloc[start_retidx]
        max_retdate = cls._parameters['returns']['date'].iloc[-end_retidx]
        dates = cls._parameters['returns']['date']
        colnames = cls._parameters['returns'].columns

        idx_filter = None
        if 'mkt_idx' in df.columns:
            idx_filter = df['mkt_idx'].astype(str).isin(colnames)
        
        df_filter = (
            (df['event_date'] >= min_retdate) 
            & (df['event_date'] <= max_retdate) 
            & (df['event_date'].isin(dates))
            & (df['ticker'].astype(str).isin(colnames)))
        
        if idx_filter is not None:
            df_filter = (df_filter) & (idx_filter)

        df = df[df_filter]
        exclude_df = df[~df_filter]
        return (df, exclude_df)

    @classmethod
    def MarketModel(
        cls,
        ticker: str,
        mkt_idx: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 30,
        weight: int = 1
    ):
        """
        Model the returns with the market model.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        mkt_idx : str
            Ticker symbol of the market index (e.g., 'SPY', '^GSPC').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the market model with the S&P500 index as a market proxy.
        >>> event = SingleEvent.MarketModel(
        ...     ticker='AAPL',
        ...     mkt_idx='SPY',
        ...     event_date=np.datetime64('2007-01-09'),
        ...     event_window=(-5,+20)
        ... )
        """
        daily_ret, daily_mkt = cls._get_parameters(
            "returns",
            (ticker, mkt_idx,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        description = f"Market model estimation, Ticker: {ticker}, Market: {mkt_idx}"

        return cls(
            MarketModel,
            {"daily_ret": daily_ret, "daily_mkt": daily_mkt},
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description= description,
            ticker=ticker,
            mkt_idx=mkt_idx,
            event_date=event_date,
            weight=weight
        )
    
    @classmethod
    def MarketAdjustedModel(
        cls,
        ticker: str,
        mkt_idx: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 30,
        weight: int = 1
    ):
        """
        Model the returns with the market adjusted model.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        mkt_idx : str
            Ticker symbol of the market index (e.g., 'SPY', '^GSPC').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the market adjusted model with the S&P500 index as a market proxy.
        >>> event = SingleEvent.MarketAdjustedModel(
        ...     ticker='AAPL',
        ...     mkt_idx='SPY',
        ...     event_date=np.datetime64('2007-01-09'),
        ...     event_window=(-5,+20)
        ... )
        """
        daily_ret, daily_mkt = cls._get_parameters(
            "returns",
            (ticker, mkt_idx,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        description = f"Market adjusted model estimation, Ticker: {ticker}, Market: {mkt_idx}"

        return cls(
            MarketAdjustedModel,
            {"daily_ret": daily_ret, "daily_mkt": daily_mkt},
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description= description,
            ticker=ticker,
            mkt_idx=mkt_idx,
            event_date=event_date,
            weight=weight
        )

    @classmethod
    def MeanAdjustedModel(
        cls,
        ticker: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the mean adjusted model.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the mean adjusted model.
        >>> event = SingleEvent.MeanAdjustedModel(
        ...     ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """
        # the comma after 'daily_ret' unpack the one-value tuple returned by the function _get_parameters
        (daily_ret,) = cls._get_parameters(
            "returns",
            (ticker,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        description = f"Mean adjusted estimation, Ticker: {ticker}"
        
        return cls(
            MeanAdjustedModel,
            {"daily_ret": daily_ret},
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            ticker=ticker,
            event_date=event_date,
            weight=weight
        )
    
    @classmethod
    def OrdinaryReturnsModel(
        cls,
        ticker: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the ordinary returns model.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.
        
        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the ordinary returns model.
        >>> event = SingleEvent.OrdinaryReturnsModel(
        ...     ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """
        # the comma after 'daily_ret' unpack the one-value tuple returned by the function _get_parameters
        (daily_ret,) = cls._get_parameters(
            "returns",
            (ticker,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        description = f"Raw Returns, Ticker: {ticker}"
        
        return cls(
            OrdinaryReturnsModel,
            {"daily_ret": daily_ret},
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            ticker=ticker,
            event_date=event_date,
            weight=weight
        )

    @classmethod
    def FamaFrench3(
        cls,
        ticker: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Fama-French 3-factor model.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.

        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the Fama-French 3-factor model.
        >>> event = SingleEvent.FamaFrench3(
        ...     ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """

        (daily_ret,) = cls._get_parameters(
            "returns",
            (ticker,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        Mkt_RF, SMB, HML, RF = cls._get_parameters(
            "FamaFrench",
            ("mkt_rf", "ff3_smb", "ff3_hml", "ff3_rf"),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )

        description = f"Fama-French 3-factor model estimation, Ticker: {ticker}"
        
        return cls(
            FamaFrench3,
            {
                "daily_ret": daily_ret,
                "Mkt_RF": Mkt_RF,
                "SMB": SMB,
                "HML": HML,
                "RF": RF,
            },
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            ticker=ticker,
            event_date=event_date,
            weight=weight
        )

    @classmethod
    def FamaFrench5(
        cls,
        ticker: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the Fama-French 5-factor model.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.

        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the Fama-French 5-factor model.
        >>> event = SingleEvent.FamaFrench5(
        ...     ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """

        (daily_ret,) = cls._get_parameters(
            "returns",
            (ticker,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        Mkt_RF, SMB, HML, RMW, CMA, RF = cls._get_parameters(
            "FamaFrench",
            ("mkt_rf", "ff5_smb", "ff5_hml", "ff5_rmw", "ff5_cma", "ff5_rf"),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )

        description = f"Fama-French 5-factor model estimation, Ticker: {ticker}"
        
        return cls(
            FamaFrench5,
            {
                "daily_ret": daily_ret,
                "Mkt_RF": Mkt_RF,
                "SMB": SMB,
                "HML": HML,
                "RMW": RMW,
                "CMA": CMA,
                "RF": RF,
            },
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            ticker=ticker,
            event_date=event_date,
            weight=weight
        )
    
    @classmethod
    def Carhart(
        cls,
        ticker: str,
        event_date: np.datetime64,
        event_weight: int = 1,
        event_window: tuple = (-5, +5),
        est_size: int = 252,
        buffer_size: int = 21,
        weight: int = 1,
        **kwargs
    ):
        """
        Model the returns with the Carhart (Fama-French 3-factor + Momentum) model.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
        event_date : np.datetime64
            Date of the event in numpy.datetime64 format.
        event_window : tuple, optional
            Event window specification (T2,T3), by default (-5, +5).
            A tuple of two integers, representing the start and the end of the event window. 
            Classically, the event-window starts before the event and ends after the event.
            For example, `event_window = (-2,+20)` means that the event-period starts
            2 periods before the event and ends 20 periods after.
        est_size : int, optional
            Size of the estimation for the modelisation of returns [T0,T1], by default 252
        buffer_size : int, optional
            Size of the buffer window [T1,T2], by default 21
        weight : int, optional
            Weight to be applied to the returns in the MultipleEvents Object
        **kwargs
            Additional keywords have no effect but might be accepted to avoid freezing 
            if there are not needed parameters specified.

        Example
        -------
        Run an event study for the Apple company for the announcement of the first iphone,
        based on the Carhart model.
        >>> event = SingleEvent.Carhart(
        ...     ticker = 'AAPL',
        ...     event_date = np.datetime64('2007-01-09'),
        ...     event_window = (-5,+20)
        ... )
        """

        (daily_ret,) = cls._get_parameters(
            "returns",
            (ticker,),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )
        Mkt_RF, SMB, HML, RF, MOM = cls._get_parameters(
            "FamaFrench",
            ("mkt_rf", "ff3_smb", "ff3_hml",  "ff3_rf", "mom"),
            event_date,
            event_weight,
            event_window,
            est_size,
            buffer_size,
            weight
        )

        description = f"Carhart model estimation, Ticker: {ticker}"
        
        return cls(
            Carhart,
            {
                "daily_ret": daily_ret,
                "Mkt_RF": Mkt_RF,
                "SMB": SMB,
                "HML": HML,
                "RF": RF,
                "MOM": MOM,
            },
            event_weight=event_weight,
            event_window=event_window,
            est_size=est_size,
            buffer_size=buffer_size,
            description=description,
            ticker=ticker,
            event_date=event_date,
            weight=weight
        )