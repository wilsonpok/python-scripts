# https://datascienceplus.com/predict-customer-churn-logistic-regression-decision-tree-and-random-forest/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import modelplotpy as mp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix


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


################
# Preprocessing
################

X, y = df.drop('Churn', axis=1), df['Churn']

X = pd.get_dummies(X)



################
# Partition data
################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666)

X_train.shape, y_train.shape
# ((4711, 42), (4711,))

X_test.shape, y_test.shape
# ((2321, 42), (2321,))



################
# Fit model
################

model = LogisticRegressionCV(cv=5, random_state=666, n_jobs=8, max_iter=500)

model_fit = model.fit(X_train, y_train)

coef_vals = pd.DataFrame(np.transpose(model_fit.coef_))
coef_vals.columns = ['values']

coefficients = pd.concat([pd.DataFrame({'coef' : X.columns}), coef_vals], axis = 1)

coefficients['abs_values'] = abs(coefficients['values'])

coefficients.sort_values('abs_values', ascending=False, inplace=True)



################
# Assess model
################

test_preds = model_fit.predict(X_test)
test_pred_probs = model_fit.predict_proba(X_test)[:,0]

confusion_matrix(y_true=y_test, y_pred=test_preds)
# array([[1566,  160],
#        [ 320,  275]])





obj = mp.modelplotpy(feature_data = [X_train, X_test]
                     , label_data = [y_train, y_test]
                     , dataset_labels = ['train data', 'test data']
                     , models = [model_fit]
                     , model_labels = ['logistic regression']
                     )


ps = obj.plotting_scope(select_model_label = ['logistic regression'], select_dataset_label = ['train data'])


