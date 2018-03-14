# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import quandl
import math, datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pickle
style.use('ggplot')

#df=quandl.get("SSE/GGQ1", authtoken="3nVssiyrfvjfvuLCETjm")
#df = df[['High','Low','Last','Previous Day Price','Volume']]
#df['HL_pct'] = (df['High']-df['Last'])/df['Last']*100
#df['pct_change'] = (df['Last']-df['Previous Day Price'])/df['Previous Day Price']*100

#df = df[['Last','HL_pct','pct_change','Volume']]

#forecast_col = 'Last'

df = quandl.get("NSE/CNX_PHARMA", authtoken="3nVssiyrfvjfvuLCETjm")
df = df[['Open','High','Low','Close','Shares Traded']]
df['HL_pct'] = (df['High']-df['Close'])/df['Close']*100
df['pct_change'] = (df['Close']-df['Open'])/df['Open']*100

df = df[['Close','HL_pct','pct_change','Shares Traded']]

forecast_col = 'Close'

df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

# print(df.head())

# x is the features (i.e. data existing) y is the new prices

x = np.array(df.drop(['label'],1)) # drop last col (label/price)
x = preprocessing.scale(x)
x_lately = x[-forecast_out:] # set future x to blank space after last predicted training value. We will need to fill this missing data in
x = x[:-forecast_out] # setting x to only be up to the filled-in predicted training data

df.dropna(inplace=True) 

y = np.array(df['label'])
#print(len(x),len(y))

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

# since we saved a pickle file, we don't need to retrain the classifier every time we run this code. It's saved the training session
#clf = LinearRegression(n_jobs=-1)
#clf.fit(x_train,y_train)
#with open('linearregression.pickle', 'wb') as f:
#    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)

# print(accuracy)

forecast_set = clf.predict(x_lately)
#print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name # get the last date of existing dataset
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day # set a new date for ongoing days

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4) # placing legend on bottom right of chart
plt.xlabel('Date')
plt.ylabel('Price')






