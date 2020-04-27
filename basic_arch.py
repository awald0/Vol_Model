import datetime as dt

import pandas as pd

from arch import arch_model
import matplotlib.pyplot as plt

sp500 = pd.read_csv('/Users/alexanderwald/Documents/spx_data copy.csv')
sp500['Datetime'] = pd.to_datetime(sp500['Date'])
sp500 = sp500.set_index('Datetime')
returns = 100 * sp500['Adj Close'].pct_change().dropna()
am = arch_model(returns)


res = am.fit()

print(res.summary())

#forecast
am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
split_date = dt.datetime(2003,12,24)
res = am.fit(last_obs=split_date)

forecasts = res.forecast(horizon=5, start=split_date)
forecast_variance = forecasts.variance[split_date:].plot()
print(forecasts.variance.tail())

forecast_mean = forecasts.mean[split_date:].plot()
plt.show(forecast_mean)
plt.show(forecast_variance)

print(forecasts.mean.iloc[-3:])
print(forecasts.residual_variance.iloc[-3:])
print(forecasts.variance.iloc[-3:])
