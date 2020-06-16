# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:39:35 2020

@author: ali hussain
"""

### Loading required libraries

import  matplotlib.pyplot  as       plt
import  statsmodels.api          as       sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.filterwarnings("ignore")
import statsmodels.tsa.api as smt
import pandas as pd
# Read the time series data
# We set parse_dates to make sure data is read as time-series and make time column as Index
# We drop the id column, which is not usefull for us.
predice=pd.read_csv('D:/predice_el_futuro/data_set/train_csv.csv',header=0,index_col=1,parse_dates=True,squeeze = True)
predice.drop(['id'],axis = 1,inplace = True)

#Let's now visualise the data

plt.figure(figsize=(12,6))
plt.plot(predice.index, predice['feature'])
plt.title('Feature v/s Time Plot')
plt.xlabel('Time')
plt.ylabel('Feature')
plt.grid(True)
plt.show()

#Let's Decompose the Graph for better understanding

decompPred = sm.tsa.seasonal_decompose(predice.feature, model="additive", freq=4)
decompPred.plot()
plt.show()

#By analysing the graphical representation of Feature v/s Time and Decomposed graph, we see only a slight upward trend in the data.
#Splitting data to Test:Train
train = predice[0:int(len(predice)*0.7)] 
test = predice[int(len(predice)*0.7):]
# extract out the time-series
feature_ts_train =  train['feature']
feature_ts_test =  test['feature']
#In this notebook we will perform ARIMA on this dataset.
#Step 1 : Perform Dicky Fuller Test on the timeseries and verify the null hypothesis that the TS is non-stationary.

dftest = adfuller(feature_ts_train)
dftest
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is %1.4f' %dftest[1])

#DF test statistic is -2.019
#DF test p-value is 0.2782
#Since p-value > 0.05. Null Hypothesis cannot be rejected.

#Let us try differencing, to make the time-series Stationary.

plt.figure(figsize=(8, 4))
plt.plot(feature_ts_train.diff(periods=1))
plt.xlabel('Time')
plt.ylabel('Feature')
#Text(0, 0.5, 'Feature')

dftest = adfuller(feature_ts_train)
dftest
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is %1.4f' %dftest[1])

#DF test statistic is -2.019
#DF test p-value is 0.2782
#The data is Stationary now, let us apply ARIMA on the differenced data set.

feature_ts_train_diff = feature_ts_train.diff(periods=1)
feature_ts_train_diff.dropna(inplace=True)

feature_ts_test_diff = feature_ts_test.diff(periods=1)
feature_ts_test_diff.dropna(inplace=True)
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
plt.xticks(range(0,30,1), rotation = 90)
smt.graphics.plot_acf(feature_ts_train_diff, lags=30, ax=axes[0])
smt.graphics.plot_pacf(feature_ts_train_diff, lags=30, ax=axes[1])
plt.tight_layout()

#Inference
#The above ACF has “decayed” fast and remains within the significance range (blue band). This is indicative of a stationary series.

#Starting to Implement ARIMA
# Build Model 
model = ARIMA(feature_ts_train_diff, order=(1, 1, 0)) #Chosen this combination of p,q,d after trying other combinations 
fitted = model.fit()  

# Forecast
fc, se, conf = fitted.forecast(24, alpha=0.05)  

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(8,5), dpi=100)
plt.plot(feature_ts_train_diff, label='training')
plt.plot(feature_ts_test_diff, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()



forecast = fitted.forecast(steps=40)[0]
predice_to_pred = pd.read_csv('D:/predice_el_futuro/data_set/test_csv.csv')
predice_to_pred['forecast'] = forecast
predice_to_pred.head()

predice_to_pred.to_csv("D:/predice_el_futuro/data_set/Solution2.csv")  





'''
data=pd.read_csv('D:/predice_el_futuro/data_set/solution3.csv')
data.head()


data['expected']
data['forecast']

from sklearn.metrics import mean_squared_error
from math import sqrt
mse = mean_squared_error(data['expected'], data['forecast'])
rmse = sqrt(mse)
print('RMSE: %f' % rmse)
'''
