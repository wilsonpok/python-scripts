# https://keshan.github.io/Churn_prediction-on-tensorflow/

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model



#################
# Load data
#################

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()

data.columns

data.dtypes



#################
# Clean data
#################

data.SeniorCitizen.replace([0, 1], ["No", "Yes"], inplace= True)


data.TotalCharges = data.TotalCharges.astype(float)

for charge in data.TotalCharges:
  try:
    charge = float(charge)
  except:
    print("charge is: %s" % charge)


for i in range(len(data)):
  if data.TotalCharges[i] == " ":
      print("Tenure is %s and Monthly charges are %s" % (data.tenure[i], data.MonthlyCharges[i]))


data.TotalCharges.replace([" "], ["0"], inplace= True)
data.TotalCharges = data.TotalCharges.astype(float)



data.drop("customerID", axis= 1, inplace= True)


for col in data.dtypes[data.dtypes == object].index:
    print(col, data[col].unique())


data.Churn.replace(["Yes", "No"], [1, 0], inplace= True)

data = pd.get_dummies(data)



#################
# Prepare data for modelling
#################

X = data.drop("Churn", axis= 1)
y = data.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)


#################
# Train model
#################

model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=10)


#################
# Assess model
#################

_, accuracy = model.evaluate(X_test, y_test)


#################
# Output model
#################

model.save('my_model.h5')
