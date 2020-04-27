import os
import sys
import datetime as dt

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl

#Data feed
import quandl
#quandl.ApiConfig.api_key = "" #api key


# TS Plots
def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return

# Simulating a GARCH(1, 1) process Not used

#np.random.seed(2)

#a0 = 0.2
#a1 = 0.5
#b1 = 0.3

#n = 10000
#w = np.random.normal(size=n)
#eps = np.zeros_like(w)
#sigsq = np.zeros_like(w)

#for i in range(1, n):
#    sigsq[i] = a0 + a1*(eps[i-1]**2) + b1*sigsq[i-1]
#    eps[i] = w[i] * np.sqrt(sigsq[i])

# Fit a GARCH(1, 1) model to our simulated EPS series
# We use the arch_model function from the ARCH package

#am = arch_model(eps)
#res = am.fit(update_freq=5)
#print(res.summary())

# Choose best GARCH model
def _get_best_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue

    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


#mydata = quandl.get('WIKI/BRK_A',rows=1001)
mydata = pd.read_csv('/Users/alexanderwald/Documents/spx_data.csv')
mydata['Datetime'] = pd.to_datetime(mydata['Date'])
mydata = mydata.set_index('Datetime')
mydata = 100 * mydata['Close'].pct_change().dropna()#
TS = mydata

optimize = raw_input("Optimize ARIMA? y/n: ")
if optimize.lower() == 'y':
    res_tup = _get_best_model(TS)
    best_order = res_tup[1]
    print best_order
else:
    print("GARCH(1,1) will be used as default")
    best_order = [1,0,1]

# Now we can fit the arch model using the best fit arima model parameters
p = best_order[0]
o = best_order[1]
q = best_order[2]

if best_order[0] + best_order[1] == 0:#fix potential eror to flip to garch(1,1)
    print "Set to GARCH(1,1) GARCH(p,o,q) must have p or q > 0"
    p = 1
    o = 0
    q = 1

split_date = dt.datetime(2002,1,1)

# Using student T distribution usually provides better fit
am = arch_model(TS, p=p, o=o, q=q, dist='StudentsT')
res = am.fit(update_freq=5, disp='off')
print(res.summary())

fig = res.plot(annualize='D')
plt.show(fig)

plot1 = tsplot(res.resid, lags=30)
plt.show(plot1)

forecasts = res.forecast(horizon=1,start=split_date)
forecasts_variance = forecasts.variance

#forecasts data
print(forecasts.mean.iloc[-3:])
print(forecasts.residual_variance.iloc[-3:])
print(forecasts.variance.iloc[-3:])
print forecast.variance.tail()

# Plot 21 day forecast for SPY vol
plt.style.use('bmh')
fig = plt.figure(figsize=(9,7))
ax = plt.gca()
ts = TS.rolling(2).std()
ts.plot(ax=ax, label='SPX Volatility')
# in sample prediction
pred = (forecasts_variance)
pred.plot(ax=ax, style='r-', label='Forecast variance')

styles = ['b-', '0.2', '0.75', '0.2', '0.75']
plt.legend(loc='best', fontsize=10)
plt.show()
