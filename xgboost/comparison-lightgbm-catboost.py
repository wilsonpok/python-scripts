#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:51:48 2018

@author: wilsonpok
"""

import xgboost as xgb
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


####################
# Load data
####################


data = pd.read_csv("~/sample-datasets/flight-delays/flights.csv")
data = data.sample(frac=0.1, random_state=10)

data = data[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
             "DESTINATION_AIRPORT", "ORIGIN_AIRPORT", "AIR_TIME",
             "DEPARTURE_TIME", "DISTANCE", "ARRIVAL_DELAY"]]

data.dropna(inplace=True)

data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"] > 10) * 1

cols = ["AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT", "ORIGIN_AIRPORT"]

for item in cols:
    data[item] = data[item].astype("category").cat.codes + 1

train, test, y_train, y_test = train_test_split(
        data.drop(["ARRIVAL_DELAY"], axis=1),
        data["ARRIVAL_DELAY"], random_state=10, test_size=0.25)


####################
# xgboost
####################

def auc(m, train, test):
    return (metrics.roc_auc_score(
            y_train, m.predict_proba(train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]))


# Parameter Tuning
model = xgb.XGBClassifier()

param_dist = {"max_depth": [10, 30, 50],
              "min_child_weight": [1, 3, 6],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1, 0.16], }

grid_search = GridSearchCV(model, param_grid=param_dist, cv=3,
                           verbose=10, n_jobs=-1)

grid_search.fit(train, y_train)

grid_search.best_estimator_

model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,
                          n_jobs=-1, verbose=1, learning_rate=0.16)

model.fit(train, y_train)

auc(model, train, test)
