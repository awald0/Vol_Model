import os
import sys
import datetime as dt

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import math

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl

#Data feed
#import quandl
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

#am = arch_model(eps)
#res = am.fit(update_freq=5)
#print(res.summary())

#Choose best GARCH model
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
mydata = pd.read_csv('/Users/alexanderwald/Dropbox/Vol_Model/spx_data.csv')
mydata['Date'] = pd.to_datetime(mydata['Date'])
mydata = mydata.set_index('Date')
mydata = 100 * mydata['Close'].pct_change().dropna()
#log caused issue so actual % chagne used
TS = mydata

optimize = raw_input("Optimize ARIMA? y/n: ")
if optimize.lower() == 'y':
    res_tup = _get_best_model(TS)
    best_order = res_tup[1]
    print best_order
else:
    print("GARCH(1,1) will be used as default")
    best_order = [1,0,1]

#fit tarch model using best fit arima model parameters
p = best_order[0]
o = best_order[1]
q = best_order[2]

if best_order[0] + best_order[1] == 0:#fix potential error to flip to garch(1,1)
    print "Set to GARCH(1,1) GARCH(p,o,q) must have p or q > 0"
    p = 1
    o = 0
    q = 1
#option for user to input split date
split_date = dt.datetime(2016,1,1)

# GARCH model using student T distribution
am = arch_model(TS, vol='Garch', p=p, o=o, q=q, dist='StudentsT')
res = am.fit(update_freq=5, disp='off')
print(res.summary())

#plot
fig1 = res.plot(annualize='D')
plt.savefig('residuals_analysis1.png')
plt.show(fig1)

#residual analysis
fig2 = tsplot(res.resid, lags=30)
plt.savefig('residuals_analysis2.png')
plt.show(fig2)

#generate forecast
#info: http://arch.readthedocs.io/en/latest/univariate/forecasting.html'
#toggle forecast type
#Simulation-based forecasts use the model random number generator to simulate draws of the standardized residuals, et+h.

forecast_period = raw_input('Enter (1m)/(3m) for 1 or 3 month forecast or enter n days: ')
if forecast_period.lower() == '1m':
    horizon = 20
elif forecast_period.lower() == '3m':
    horizon = 20*3
elif int(forecast_period) > 120:
    print('Please choose a lower forecast period!')
else:
    horizon = int(forecast_period)

forecast_type = raw_input('Switch to Simulation forecast? Default is Analytic: y/n: ')
if forecast_type.lower() == 'y':
    forecasts = res.forecast(horizon=horzion, start=split_date, method='simulation')
else:
    forecasts = res.forecast(horizon=horizon,start=split_date)
forecasts_variance = forecasts.variance

#save forecast
forecasts.variance.to_csv('forecast_volatility.csv', sep='\t')

#print forecasts data
print(forecasts.mean.iloc[-3:])
print(forecasts.residual_variance.iloc[-3:])
print(forecasts.variance.iloc[-3:])
print('Next N days forecast variance:')
last_forecasts = res.forecast(horizon=horizon,start=split_date)
print(last_forecasts.variance.iloc[-1:])

#plot forecasts
plt.style.use('bmh')
fig3 = plt.figure(figsize=(9,7))
ax = plt.gca()
#ts = TS.rolling(10).std()*math.sqrt(252/10)
#ts.plot(ax=ax, label='SPX Volatility')

#in sample prediction
forecast_col = ('h.%s' % horizon)
if horizon < 10:
    forecast_col = ('h.0%s' % horizon)
else:
    forecast_col = ('h.%s' % horizon)
#in sample prediction
forecasts_variance[forecast_col] = np.sqrt(forecasts_variance[forecast_col]*252)
pred = (forecasts_variance[forecast_col])
pred.plot(ax=ax, style='r-', label='Forecast variance: %s day' % horizon)

#load in vix futures
vx1 = pd.read_csv('/Users/alexanderwald/Dropbox/Vol_Model/CHRIS-CBOE_VX1.csv')
vx1['Trade Date'] = pd.to_datetime(vx1['Trade Date'])
vx1 = vx1.set_index('Trade Date')
vx1 = vx1['Close']
vx1 = vx1[vx1!=0].loc[:split_date]
vx1.plot(ax=ax, style='b-', label='VX1 Future')

#spot vix
#vix1 = pd.read_csv('/Users/alexanderwald/Documents/VIX_data.csv')
#vix1['DATE'] = pd.to_datetime(vix1['DATE'])
#vix1 = vix1.set_index('DATE')
#vix1 = vix1['VIXCLS']
#vix1 = vix1[vx1!="."].loc[:split_date]
#vix1.plot(ax=ax, style='g-', label='VX1 Future')

styles = ['b-', '0.2', '0.75', '0.2', '0.75']
plt.legend(loc='best', fontsize=10)
plt.savefig('forecast_variance.png')
plt.show()
