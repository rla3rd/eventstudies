Eventstudies Library

Large portions of this library were lifted from
https://github.com/LemaireJean-Baptiste/eventstudy
and updated.  It is really just a fancy pants wrapper around
statsmodels.api.OLS function, which is an ordinary least squares regression
function.

The general premise of event studies is to do a regression analysis of some
period prior to the event in question to come up with a baseline equation
to estimate the log returns of a single event and determine if the actual return
is statistically significant compared to the prediction.  This leads to a
abnormal return value (AR), this value is the return above the expected return
predicted + prediction error from the regression from the prior period. The
daily AR values are then summed over the event window to creare a cumulative
abnormal return value (CAR).  A t-test statistic is then generated over the
event window for each day's CAR value, and the P-value is then calculated
for significance daily over the event window to determine if the CAR values
have any statistical significance that indicate that these values are indeed
different from what was expected.

The single events as described above are then aggregated and analyzed using
the same t-test and P-value methodologies to determine if the aggregate of
all events of the same type are statistically significant. These values are
refered to as the average abnormal return (AAR) and the cumulative average
abnormal return (CAAR).

This package has the following models: MeanAdjustedModel, RawReturnsModel,
MarketAdjustedModel, MarketModel, FamaFrench3, Carhart, and FamaFrench5.  

The models for each single event are described below as follows.

MeanAdjustedModel

The mean daily return is calculated for the est_size period, the AR value is
daily returns - the mean daily return.  The CAR value is the cumulative
sum of (AR).

RawReturnsModel

AR is just the security's daily returns. The CAR value is the cumulative 
sum of (AR).

MarketAdjustedModel

The AR value is the daily returns - daily returns of a given market index.
The CAR value is the cumulative sum of (AR).

MarketModel

A OLS regression is performed between a security's daily returns and the  
daily returns of a given market index over the est_size period.  Alpha and
beta are calculated from the regression.  The AR value is the daily returns
- (predicted value of returns + std error of prediction) using the OLS
regression results. The CAR value is the cumulative sum of (AR).

FamaFrench3

A OLS regression is performed between a security's daily returns vs the Fama
French 3 factors over the est_size period.  Alpha and beta are calculated 
from the regression.  The AR value is the daily returns - (predicted value of
returns + std error of prediction) using the OLS regression results. 
The CAR value is the cumulative sum of (AR).

Carhart

A OLS regression is performed between a security's daily returns vs the Fama
Carhart Factors (FamaFrench3 + Momentum) over the est_size period.  Alpha and
beta are calculated from the regression.  The AR value is the daily returns
- (predicted value of returns + std error of prediction) using the OLS regression
results. The CAR value is the cumulative sum of (AR).

FamaFrench5

A OLS regression is performed between a security's daily returns vs the Fama
French 5 factors over the est_size period.  Alpha and beta are calculated 
from the regression.  The AR value is the daily returns - (predicted value of
returns + std error of prediction) using the OLS regression results.
The CAR value is the cumulative sum of (AR).
