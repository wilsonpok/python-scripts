# https://datascienceplus.com/predict-customer-churn-logistic-regression-decision-tree-and-random-forest/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



################
# Load data
################

filename = os.path.expanduser('~/data/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df = pd.read_csv(filename).set_index('customerID')

df.head()

df.shape
# (7043, 20)

df.columns.tolist()

df.describe()




################
# Clean data
################

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(inplace=True)

df.shape
# (7032, 20)

df['OnlineSecurity'].loc[df.OnlineSecurity == 'No internet service'] = 'No'
df['OnlineBackup'].loc[df.OnlineBackup == 'No internet service'] = 'No'
df['DeviceProtection'].loc[df.DeviceProtection == 'No internet service'] = 'No'
df['TechSupport'].loc[df.TechSupport == 'No internet service'] = 'No'
df['StreamingTV'].loc[df.StreamingTV == 'No internet service'] = 'No'
df['StreamingMovies'].loc[df.StreamingMovies == 'No internet service'] = 'No'
df['MultipleLines'].loc[df.MultipleLines == 'No phone service'] = 'No'



################
# Pre-process data
################

# Tenure groups
df['tenure'].min()
# 1

df['tenure'].max()
# 72

df.loc[df.tenure <= 12, 'group_tenure'] = '0–12 month'
df.loc[(df.tenure > 12) & (df.tenure <= 24), 'group_tenure'] = '13–24 month'
df.loc[(df.tenure > 24) & (df.tenure <= 48), 'group_tenure'] = '25–48 month'
df.loc[(df.tenure > 48) & (df.tenure <= 60), 'group_tenure'] = '49–60 month'
df.loc[df.tenure > 60, 'group_tenure'] = '60+ month'

df.drop('tenure', inplace=True, axis=1)


# Senior citizen
df.loc[df.SeniorCitizen == 1, 'SeniorCitizen'] = 'Yes'
df.loc[df.SeniorCitizen == 0, 'SeniorCitizen'] = 'No'



################
# Explore data
################


# Numeric
df.dtypes

df_numeric = df[['MonthlyCharges', 'TotalCharges']]

df_numeric.corr()
#                 MonthlyCharges  TotalCharges
# MonthlyCharges        1.000000      0.651065
# TotalCharges          0.651065      1.000000

df.drop('TotalCharges', inplace=True, axis=1)


# Categorical
plt.close('all')
sns.countplot(x='gender', data=df)
plt.show()

plt.close('all')
sns.countplot(x='SeniorCitizen', data=df)
plt.show()

plt.close('all')
sns.countplot(x='Partner', data=df)
plt.show()

plt.close('all')
sns.countplot(x='Dependents', data=df)
plt.show()

plt.close('all')
sns.countplot(x='PhoneService', data=df)
plt.show()

plt.close('all')
sns.countplot(x='MultipleLines', data=df)
plt.show()

plt.close('all')
sns.countplot(x='InternetService', data=df)
plt.show()

plt.close('all')
sns.countplot(x='OnlineSecurity', data=df)
plt.show()

plt.close('all')
sns.countplot(x='OnlineBackup', data=df)
plt.show()

plt.close('all')
sns.countplot(x='DeviceProtection', data=df)
plt.show()

plt.close('all')
sns.countplot(x='TechSupport', data=df)
plt.show()

plt.close('all')
sns.countplot(x='StreamingTV', data=df)
plt.show()

plt.close('all')
sns.countplot(x='StreamingMovies', data=df)
plt.show()

plt.close('all')
sns.countplot(x='PaperlessBilling', data=df)
plt.show()

plt.close('all')
sns.countplot(x='PaymentMethod', data=df)
plt.show()

plt.close('all')
sns.countplot(x='group_tenure', data=df)
plt.show()











# Scale

# Train test split

# Pipeline

# Cross validation

# modelplotpy
