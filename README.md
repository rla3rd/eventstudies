# Event Studies Library

A Python package for performing financial event study analyses. This library provides tools to analyze the impact of events on security returns using various statistical models. This package is a heavily modified version of Jean-Baptiste Lemaire's eventstudy package (https://github.com/LemaireJean-Baptiste/eventstudy)

## Overview

The general premise of event studies is to perform a regression analysis of some period prior to the event in question to establish a baseline equation that estimates the log returns of a single event and determines if the actual return is statistically significant compared to the prediction. This leads to an abnormal return value (AR), which is the return above the expected return (predicted value + prediction error from the regression from the prior period). The daily AR values are then summed over the event window to create a cumulative abnormal return value (CAR). A t-test statistic is generated over the event window for each day's CAR value, and the P-value is calculated for significance daily over the event window to determine if the CAR values have any statistical significance that indicates these values are indeed different from what was expected.

Single events as described above are then aggregated and analyzed using the same t-test and P-value methodologies to determine if the aggregate of all events of the same type are statistically significant. These values are referred to as the average abnormal return (AAR) and the cumulative average abnormal return (CAAR).

## Installation

```bash
pip install eventstudies
```

Or install from source:

```bash
git clone https://github.com/rla3rd/eventstudies.git
cd eventstudies/eventstudies
pip install -e .
```

## Requirements

- Python 3.10 or higher
- numpy >= 1.26.0
- pandas >= 2.2.3
- scipy >= 1.12.0
- statsmodels >= 0.14.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- pandas_datareader >= 0.10.0 (for fetching market data)

## Quick Start

```python
import eventstudies as es
from eventstudies import SingleEvent, MultipleEvents
from eventstudies.util import create_returns_parquet
import numpy as np

# Option 1: Create returns parquet from pandas_datareader
create_returns_parquet(
    tickers=['AAPL', 'GOOGL', 'MSFT', 'SPY'],
    output_path='data/market/returns.parquet',
    start_date='2020-01-01',
    end_date='2023-12-31'
)
SingleEvent.import_returns(path='data/market/returns.parquet')

# Option 2: Import returns from existing file
# SingleEvent.import_returns('returns.csv')

# Import Fama-French factors
SingleEvent.import_FamaFrench()

# Run a single event study
event = SingleEvent.MarketModel(
    ticker='AAPL',
    mkt_idx='SPY',
    event_date=np.datetime64('2023-01-15'),
    event_window=(-5, +10),
    est_size=252,
    buffer_size=21
)

# Display results
print(event.results())
event.plot()
```

## Available Models

This package provides the following event study models:

- **MeanAdjustedModel**: The mean daily return is calculated for the `est_size` period. The AR value is daily returns minus the mean daily return. The CAR value is the cumulative sum of AR.

- **OrdinaryReturnsModel** (RawReturnsModel): AR is just the security's daily returns. The CAR value is the cumulative sum of AR.

- **MarketAdjustedModel**: The AR value is the daily returns minus daily returns of a given market index. The CAR value is the cumulative sum of AR.

- **MarketModel**: An OLS regression is performed between a security's daily returns and the daily returns of a given market index over the `est_size` period. Alpha and beta are calculated from the regression. The AR value is the daily returns minus (predicted value of returns + std error of prediction) using the OLS regression results. The CAR value is the cumulative sum of AR.

- **FamaFrench3**: An OLS regression is performed between a security's daily returns vs the Fama-French 3 factors over the `est_size` period. Alpha and beta are calculated from the regression. The AR value is the daily returns minus (predicted value of returns + std error of prediction) using the OLS regression results. The CAR value is the cumulative sum of AR.

- **Carhart**: An OLS regression is performed between a security's daily returns vs the Carhart factors (FamaFrench3 + Momentum) over the `est_size` period. Alpha and beta are calculated from the regression. The AR value is the daily returns minus (predicted value of returns + std error of prediction) using the OLS regression results. The CAR value is the cumulative sum of AR.

- **FamaFrench5**: An OLS regression is performed between a security's daily returns vs the Fama-French 5 factors over the `est_size` period. Alpha and beta are calculated from the regression. The AR value is the daily returns minus (predicted value of returns + std error of prediction) using the OLS regression results. The CAR value is the cumulative sum of AR.

## Documentation

For detailed documentation, see the [docs](docs-src/source/index.md) directory.

## License

GNU General Public License v3 (GPLv3)

## Author

Rick Albright  
Jean-Baptiste Lemaire