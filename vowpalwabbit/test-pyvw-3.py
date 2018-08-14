import os
import zipfile
import pandas as pd
from vowpalwabbit import pyvw
import numpy as np
from sklearn.model_selection import train_test_split


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

df.loc[df.target_control == 'target', 'target_control'] = 1
df.loc[df.target_control == 'control', 'target_control'] = 2

df.loc[(df.target_control == 1) & (df.outcome == 'positive'), 'reward'] = 100
df.loc[(df.target_control == 2) & (df.outcome == 'positive'), 'reward'] = 50
df.loc[(df.target_control == 1) & (df.outcome == 'negative'), 'reward'] = -100
df.loc[(df.target_control == 2) & (df.outcome == 'negative'), 'reward'] = -50

df.loc[:, 'cost'] = -df.reward

df['reward_weighted'] = df.reward / df.prob_target / 2




#------------------
# Split train test
#------------------

train, test = train_test_split(df, test_size=0.2)



#------------------
# Train model
#------------------

vw = pyvw.vw('--cb 2')

for i in train.index:
  action = train.loc[i, 'target_control']
  cost = train.loc[i, 'cost']
  probability = train.loc[i, 'prob_target']
  x1 = train.loc[i, 'Node1']
  x2 = train.loc[i, 'Node2']
  x3 = train.loc[i, 'Node3']
  x4 = train.loc[i, 'Node4']
  x5 = train.loc[i, 'Node5']
  x6 = train.loc[i, 'Node6']
  x7 = train.loc[i, 'Node7']
  x8 = train.loc[i, 'Node8']
  x9 = train.loc[i, 'Node9']
  x10 = train.loc[i, 'Node10']
  x11 = train.loc[i, 'Node11']
  x12 = train.loc[i, 'Node12']
  x13 = train.loc[i, 'Node13']
  x14 = train.loc[i, 'Node14']
  x15 = train.loc[i, 'Node15']
  x17 = train.loc[i, 'Node17']
  x18 = train.loc[i, 'Node18']
  x19 = train.loc[i, 'Node19']
  x20 = train.loc[i, 'Node20']
  vw_instance = str(action) + ':' + str(cost) + ':'\
  + str(probability) + ' | '\
  + str(x1) + ' ' + str(x2) + ' ' + str(x3) + ' ' + str(x4) + ' ' + str(x5)\
  + ' ' + str(x6) + ' ' + str(x7) + ' ' + str(x8) + ' ' + str(x9)\
  + ' ' + str(x10)+ ' ' + str(x11)+ ' ' + str(x12)+ ' ' + str(x13)\
  + ' ' + str(x14)+ ' ' + str(x15)+ ' ' + str(x17)+ ' ' + str(x18)\
  + ' ' + str(x19)+ ' ' + str(x20)
  print(vw_instance)
  vw.learn(vw_instance)





#------------------
# Predict test set
#------------------

appended_data = []

for j in test.index:
  x1 = test.loc[j, 'Node1']
  x2 = test.loc[j, 'Node2']
  x3 = test.loc[j, 'Node3']
  x4 = test.loc[j, 'Node4']
  x5 = test.loc[j, 'Node5']
  x6 = test.loc[j, 'Node6']
  x7 = test.loc[j, 'Node7']
  x8 = test.loc[j, 'Node8']
  x9 = test.loc[j, 'Node9']
  x10 = test.loc[j, 'Node10']
  x11 = test.loc[j, 'Node11']
  x12 = test.loc[j, 'Node12']
  x13 = test.loc[j, 'Node13']
  x14 = test.loc[j, 'Node14']
  x15 = test.loc[j, 'Node15']
  x17 = test.loc[j, 'Node17']
  x18 = test.loc[j, 'Node18']
  x19 = test.loc[j, 'Node19']
  x20 = test.loc[j, 'Node20']
  choice = vw.predict('| ' + str(x1) + ' ' + str(x2) + ' ' + str(x3)\
  + ' ' + str(x4) + ' ' + str(x5) + ' ' + str(x6) + ' ' + str(x7)\
  + ' ' + str(x8) + ' ' + str(x9) + ' ' + str(x10) + ' ' + str(x11)\
  + ' ' + str(x12) + ' ' + str(x13) + ' ' + str(x14) + ' ' + str(x15)\
  + ' ' + str(x17) + ' ' + str(x18) + ' ' + str(x19) + ' ' + str(x20))
  appended_data.append(choice)

test.loc[:, 'recommendation'] = appended_data

test.groupby(['recommendation']).size() / len(test)


#------------------
# Evaluate
#------------------

test.loc[test.recommendation == 1].groupby(['customer_type']).size() /\
len(test.loc[test.recommendation == 1])
# lost_cause      0.243072
# persuadable     0.266825
# sleeping_dog    0.254157
# sure_thing      0.235946

test.loc[test.recommendation == 2].groupby(['customer_type']).size() /\
len(test.loc[test.recommendation == 2])
# lost_cause      0.261872
# persuadable     0.225237
# sleeping_dog    0.237449
# sure_thing      0.275441


sum(test.reward) / len(test)
# 1.35

sum(test.loc[test.target_control == test.recommendation, 'reward_weighted']) / len(test)
# 1.495197377535658

sum(test.loc[test.target_control == test.recommendation, 'reward']) / len(test)
# 1.5
