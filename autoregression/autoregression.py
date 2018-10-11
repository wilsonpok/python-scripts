#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:33:39 2018
@author: wilsonpok

https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from os.path import expanduser
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

file = expanduser('~') + \
    '/python-scripts/autoregression/daily-minimum-temperatures.csv'

temps = pd.read_csv(file, sep=',', parse_dates=[0])

print(temps.dtypes)
print(temps.describe())
print(temps.head())
print(temps.tail())

sns.lineplot(data=temps, x='date', y='min_temp')
plt.show()

# --------------

lag_plot(temps['min_temp'])
plt.show()

# --------------

# create lagged dataset
dataframe = temps.copy()
dataframe['min_temp_p1'] = temps.min_temp.shift(1)

result = dataframe[['min_temp', 'min_temp_p1']].corr()
print(result)

# --------------

autocorrelation_plot(temps['min_temp'])
plt.show()

# --------------

plot_acf(temps['min_temp'], lags=31)
plt.show()

# --------------
# split into train and test sets

dataframe.set_index('date', inplace=True)

X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


# persistence model
def model_persistence(x):
    return x


# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)

test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions vs expected
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()

# --------------
# split dataset

X = dataframe.reset_index()
X = X['min_temp']
train, test = X[1:len(X)-7], X[len(X)-7:]

# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1,
                                dynamic=False)

comparison = pd.concat([predictions, test], axis=1)
comparison.columns = ['predicted', 'expected']
print(comparison)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# --------------

window = model_fit.k_ar
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train)-window:].reset_index(drop=True)
history = [history[i] for i in range(len(history))]

predictions = list()

for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test.reset_index(drop=True)[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
plt.plot(test.reset_index(drop=True))
plt.plot(predictions, color='red')
plt.show()
