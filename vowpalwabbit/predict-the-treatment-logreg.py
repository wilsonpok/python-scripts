import os
import zipfile
import pandas as pd
from sklearn import linear_model

#------------------
# Load data
#------------------

os.system('wget http://ftp.cs.wisc.edu/machine-learning/shavlik-group/kuusisto.ecml14.simcustomerdata.zip')

zf = zipfile.ZipFile('kuusisto.ecml14.simcustomerdata.zip')

df = pd.read_csv(zf.open('stereotypical_customer_simulation.csv'),\
  index_col=['customer_id'])


df['target_control'].value_counts(normalize=True)
# control    0.5063
# target     0.4937

df['outcome'].value_counts(normalize=True)
# negative    0.5047
# positive    0.4953

df['customer_type'].value_counts(normalize=True)
# lost_cause      0.2554
# sleeping_dog    0.2528
# persuadable     0.2471
# sure_thing      0.2447

df.groupby(['target_control', 'outcome']).size() / len(df)
# target_control  outcome
# control         negative    0.2550
#                 positive    0.2513
# target          negative    0.2497
#                 positive    0.2440

agg = df.groupby(['customer_type', 'target_control', 'outcome']).size() /\
len(df)

print(agg)
# customer_type  target_control  outcome
# lost_cause     control         negative    0.1298
#                target          negative    0.1256
# persuadable    control         negative    0.1252
#                target          positive    0.1219
# sleeping_dog   control         positive    0.1287
#                target          negative    0.1241
# sure_thing     control         positive    0.1226
#                target          positive    0.1221


#------------------
# Process data
#------------------

df.loc[df.target_control == 'target', 'prob_target'] = 0.5063
df.loc[df.target_control == 'control', 'prob_target'] = 1 - 0.5063

df.loc[(df.target_control == 'target') & (df.outcome == 'positive'), 'reward'] = 100
df.loc[(df.target_control == 'control') & (df.outcome == 'positive'), 'reward'] = 50
df.loc[(df.target_control =='target') & (df.outcome == 'negative'), 'reward'] = -100
df.loc[(df.target_control == 'control') & (df.outcome == 'negative'), 'reward'] = -50

df.loc[:, 'cost'] = -df.reward

df['reward_weighted'] = df.reward / df.prob_target / 2



#----------------------
# Predict the treatment
# on positive instances
#----------------------

pos_instances = df.loc[df.outcome == 'positive']

logreg = linear_model.LogisticRegression(C=1e5)

filter_col = [col for col in df if col.startswith('Node')]

Y = pos_instances.target_control

X_raw = pos_instances[filter_col]

X = pd.get_dummies(X_raw)

logreg.fit(X, Y)



#----------------------
# Predict the treatment
# on all instances
#----------------------

X_all = pd.get_dummies(df[filter_col])

df['pred'] = logreg.predict(X_all)

df.groupby(['pred']).size() / len(df)


#------------------
# Evaluate
#------------------

df.loc[df.pred == 'target'].groupby(['customer_type']).size() /\
len(df.loc[df.pred == 'target'])
# lost_cause      0.242212
# persuadable     0.299640
# sleeping_dog    0.211274
# sure_thing      0.246874

df.loc[df.pred == 'control'].groupby(['customer_type']).size() /\
len(df.loc[df.pred == 'control'])
# lost_cause      0.267184
# persuadable     0.200151
# sleeping_dog    0.289907
# sure_thing      0.242757


sum(df.reward) / len(df)
# -0.755

sum(df.loc[df.target_control == df.pred, 'reward_weighted']) / len(df)
# 3.2613657744303373

sum(df.loc[df.target_control == df.pred, 'reward']) / len(df)
# 3.28



