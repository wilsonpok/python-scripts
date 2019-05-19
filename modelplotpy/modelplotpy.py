# https://modelplot.github.io/intro_modelplotpy.html

import sys
import modelplotpy as mp

print(sys.version)



import io
import os
import requests
import zipfile
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
# we encountered that the source at uci.edu is not always available, therefore we made a copy to our repos.
r = requests.get('https://modelplot.github.io/img/bank-additional.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
# You can change the path, currently the data is written to the working directory
path = os.getcwd()
z.extractall(path)
bank = pd.read_csv(path + "/bank-additional/bank-additional-full.csv", sep = ';')

# select the 6 columns
bank = bank[['y', 'duration', 'campaign', 'pdays', 'previous', 'euribor3m']]
# rename target class value 'yes' for better interpretation
bank.y[bank.y == 'yes'] = 'term deposit'

# dimensions of the data
print(bank.shape)

# show the first rows of the dataset
print(bank.head())



# to create predictive models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# define target vector y
y = bank.y
# define feature matrix X
X = bank.drop('y', axis = 1)

# Create the necessary datasets to build models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2018)

# Instantiate a few classification models
clf_rf = RandomForestClassifier().fit(X_train, y_train)
clf_mult = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg').fit(X_train, y_train)



obj = mp.modelplotpy(feature_data = [X_train, X_test]
                     , label_data = [y_train, y_test]
                     , dataset_labels = ['train data', 'test data']
                     , models = [clf_rf, clf_mult]
                     , model_labels = ['random forest', 'multinomial logit']
                     )

# transform data generated with prepare_scores_and_deciles into aggregated data for chosen plotting scope 
ps = obj.plotting_scope(select_model_label = ['random forest'], select_dataset_label = ['test data'])




# plot the cumulative gains plot
mp.plot_cumgains(ps)

# plot the cumulative gains plot and annotate the plot at decile = 2
mp.plot_cumgains(ps, highlight_decile = 2)

# plot the cumulative lift plot and annotate the plot at decile = 2
mp.plot_cumlift(ps, highlight_decile = 2)

# plot the response plot and annotate the plot at decile = 2
mp.plot_response(ps, highlight_decile = 1)

# plot the cumulative response plot and annotate the plot at decile = 3
mp.plot_cumresponse(ps, highlight_decile = 3)

# plot all four evaluation plots and save to file
mp.plot_all(ps, save_fig = True, save_fig_filename = 'Selection model Term Deposits')


# set plotting scope to model comparison
ps2 = obj.plotting_scope(scope = "compare_models", select_dataset_label=['test data'])

# plot the cumulative response plot and annotate the plot at decile = 3
mp.plot_cumresponse(ps2, highlight_decile = 3)
