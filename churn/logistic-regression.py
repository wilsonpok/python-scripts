# https://datascienceplus.com/predict-customer-churn-logistic-regression-decision-tree-and-random-forest/

import os
import pandas as pd


################
# Load data
################

filename = os.path.expanduser('~/data/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df = pd.read_csv(filename)

df.head()

df.shape
# (7043, 21)

df.columns.tolist()

df.describe()


################
# Clean data
################

df.dropna(inplace=True)

df['OnlineSecurity'].loc[df.OnlineSecurity == 'No internet service'] = 'No'
df['OnlineBackup'].loc[df.OnlineBackup == 'No internet service'] = 'No'
df['DeviceProtection'].loc[df.DeviceProtection == 'No internet service'] = 'No'
df['TechSupport'].loc[df.TechSupport == 'No internet service'] = 'No'
df['StreamingTV'].loc[df.StreamingTV == 'No internet service'] = 'No'
df['StreamingMovies'].loc[df.StreamingMovies == 'No internet service'] = 'No'


################
# Pre-process data
################


# Scale

# Train test split

# Pipeline

# Cross validation

# modelplotpy
