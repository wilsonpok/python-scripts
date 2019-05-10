#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

# load and plot dataset
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


os.chdir('/Users/wilsonpok/python-scripts/lstm-forecasting/')
os.getcwd()
os.listdir()

############################
# Shampoo Sales Dataset
############################


def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')


# load dataset
series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0],
                     index_col=0, squeeze=True, date_parser=parser)

# summarize first few rows
print(series.head())

# line plot
series.plot()
pyplot.show()


############################
# Experimental Test Setup
############################

# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]


##############################
# Persistence Model Forecast
##############################

history = [x for x in train]
predictions = list()

for i in range(len(test)):
    # make prediction
    predictions.append(history[-1])
    # observation
    history.append(test[i])


# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)

# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()


##############################
# LSTM Data Preparation
##############################

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# transform to supervised learning
X = series.values
supervised = timeseries_to_supervised(X, 1)
print(supervised.head())


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# transform to be stationary
differenced = difference(series, 1)
print(differenced.head())

# invert transform
inverted = list()

for i in range(len(differenced)):
    value = inverse_difference(series, differenced[i], len(series)-i)
    inverted.append(value)

inverted = pd.Series(inverted)
print(inverted.head())




