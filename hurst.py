# -*- coding: utf-8 -*-
"""
Compute hurst exponent for given ts
H > 0.5 trend
H = 0.5 BRW
H < 0.5 mean reversion

Input: Stock ticker
Example: hurst("SPY")

Output: Hurst exponent

@author: vincentole
"""

import os
from numpy import log, sqrt, std, subtract, reshape, array
from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web
from datetime import datetime

def hurst_stock(ticker):
    """Returns Hurst exponent from stock ticker"""
    stock = web.DataReader(ticker, "av-daily-adjusted", datetime(2000,1,1), datetime.today(), api_key=os.getenv('ALPHAVANTAGE_API_KEY'))
    ts = array(stock["adjusted close"])
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std( subtract(ts[lag:], ts[:-lag]) )) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    lm = LinearRegression().fit( log(lags).reshape(-1, 1), y = log(tau) )
    # Return the Hurst exponent
    return (lm.coef_ *2.0)[0]

def hurst_ts(ts):
    """Returns Hurst exponent from TS"""
    ts = array(ts)
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std( subtract(ts[lag:], ts[:-lag]) )) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    lm = LinearRegression().fit( log(lags).reshape(-1, 1), y = log(tau) )
    # Return the Hurst exponent
    return (lm.coef_ *2.0)[0]
