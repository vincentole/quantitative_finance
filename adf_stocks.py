# -*- coding: utf-8 -*-
"""
Compute ADF test for given tickers
H0: unit root exists
HA: mean reversion exists

Input: list of ticker symbols
Example: adf_stocks(["AAPL", "MSFT", "DAX", "SPY"])

Output: ADF statistics

@author: vincentole
"""

import os
import statsmodels.tsa.stattools as tsa
from datetime import datetime
import pandas_datareader.data as web

def adf_stocks(tickers):
    # Download data
    stocks = dict()
    for ticker in tickers:
        print("Loading %s" % ticker)
        stocks[ticker] = web.DataReader(ticker, "av-daily-adjusted", datetime(2000,1,1), datetime.today(), api_key=os.getenv('ALPHAVANTAGE_API_KEY'))
    
    # Output the results of the Augmented Dickey-Fuller test for Amazo
    # with a lag order value of 1
    print("Computing ADF")
    output = dict()
    p_values = dict()
    
    for ticker in tickers:
        adf = tsa.adfuller( stocks[ticker]["adjusted close"] )

        print("%s ADF p-value: %s" % (ticker, adf[1]))
        p_values["%s ADF p-value" % ticker] = adf[1] 
        output["%s ADF" % ticker] = adf 
    
    output["p-values"] = p_values    
    
    return output
