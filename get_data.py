import quandl
quandl.ApiConfig.api_key = "jX2sTEiWu8qUy7Rjfe4f" #api key
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data time frame
start_date = start_date=(2000, 1, 1)
start_date = datetime.date(*start_date)
end_date = start_date=(2017, 1, 1)
end_date = datetime.date(*end_date)
#Security
#mydata = quandl.get('WIKI/AAPL',rows=1001)

#print mydata.head()
#returns = mydata['Adj. Close'].pct_change().dropna()
#window = 30
#std_dev = returns.rolling(window).std()


#mydata = mydata['Adj. Close']
#TS = np.log(mydata/mydata.shift(1)).dropna() #using the log returns
TS = pd.read_csv('/Users/alexanderwald/Documents/spx_data.csv')


print TS.head()
