# -*- coding: utf-8 -*-
"""
Visualize FX

@author: vincentole
"""

import os
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

fx_tickers = ('EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF')
fx = dict()

for ticker in fx_tickers:
    print('Loading %s' % ticker)
    fx['%s' % ticker] = web.DataReader(ticker, "av-forex-daily", datetime(
        2000, 1, 1), datetime.today(), api_key=os.getenv('ALPHAVANTAGE_API_KEY'))


adj_close = pd.DataFrame(index=fx.get(fx_tickers[0])['close'].index)

for pair in fx_tickers:
    adj_close = pd.merge(
        adj_close, fx[pair]['close'], how='outer', left_index=True, right_index=True)

adj_close.columns = fx_tickers
adj_close = adj_close.pct_change().iloc[-250:] * 100


# Set seaborn theme
sns.set()

# Pairs Plot with correlations
def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

g = sns.pairplot(adj_close, kind='reg', corner=True, plot_kws={'line_kws':{'color':'red'}})
g.map_lower(corrfunc)
plt.show()


# Last returns
adj_close_long = adj_close.iloc[-30:]
adj_close_long = pd.melt(adj_close_long.reset_index(), id_vars = 'index')
means = adj_close.mean(axis = 0)

sns.set(font_scale=3)

g = sns.FacetGrid(adj_close_long, row = 'variable', height = 5, aspect = 5, sharey = True, legend_out = True)
g.map(sns.barplot, 'index', 'value', label = 'Pct Change in %')

axes = g.axes.flatten()
for i in range(0, len(axes)):
    axes[i].axhline(0, lw = 2, c = 'black')
    axes[i].axhline(means[i], ls = '-', lw = 2, c = 'red', label = 'n Day Mean')

