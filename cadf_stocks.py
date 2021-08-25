# -*- coding: utf-8 -*-
"""
Compute CADF test for given tickers
H0: not cointegrated
HA: cointegrated

Input: list of two ticker symbols
Example: cadf_stocks(["DAX", "SPY"])

Output: CADF statistics and plots

@author: vincentole
"""

import os
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
import seaborn as sns
import matplotlib.pyplot as plt




def cadf_stocks(tickers):
    # Get Data
    stocks = dict()
    for ticker in tickers:
        print("Loading %s" % ticker)
        stocks[ticker] = web.DataReader(ticker, "av-daily-adjusted", datetime(2000,1,1), datetime.today(), api_key=os.getenv('ALPHAVANTAGE_API_KEY'))
    
    # Ensure same timeframe of both ts
    x = stocks[tickers[0]]["adjusted close"]
    y = stocks[tickers[1]]["adjusted close"]
    df = pd.merge(x, y, how = "inner", left_index = True, right_index = True)
    df.columns = [tickers[0], tickers[1]]
    
    # ADF of residuals
    lm = sm.OLS(df[tickers[0]], df[tickers[1]]).fit()
    cadf = tsa.adfuller(lm.resid)
    print("CADF p-value: %s" % cadf[1])
    
    # Plot series
    sns.set()
    plt.rcParams["figure.dpi"] = 300
    
    df.plot(title = "TS Stocks")
    plt.xticks(rotation=25)
    plt.show()
    
    lm.resid.plot(title = "TS Residuals")
    plt.xticks(rotation=25)
    plt.show()
    
    return cadf
