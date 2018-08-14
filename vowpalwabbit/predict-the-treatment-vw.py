import os
import zipfile
import pandas as pd
from vowpalwabbit import pyvw


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

df.loc[df.target_control == 'target', 'target_control'] = 1.0
df.loc[df.target_control == 'control', 'target_control'] = -1.0


df.loc[:, 'cost'] = -df.reward

df['reward_weighted'] = df.reward / df.prob_target / 2



#----------------------
# Predict the treatment
# on positive instances
#----------------------

pos_instances = df.loc[df.outcome == 'positive']

vw = pyvw.vw('--loss_function logistic --binary --l2 0.5')

for i in pos_instances.index:
  label = pos_instances.loc[i, 'target_control']
  x1 = pos_instances.loc[i, 'Node1']
  x2 = pos_instances.loc[i, 'Node2']
  x3 = pos_instances.loc[i, 'Node3']
  x4 = pos_instances.loc[i, 'Node4']
  x5 = pos_instances.loc[i, 'Node5']
  x6 = pos_instances.loc[i, 'Node6']
  x7 = pos_instances.loc[i, 'Node7']
  x8 = pos_instances.loc[i, 'Node8']
  x9 = pos_instances.loc[i, 'Node9']
  x10 = pos_instances.loc[i, 'Node10']
  x11 = pos_instances.loc[i, 'Node11']
  x12 = pos_instances.loc[i, 'Node12']
  x13 = pos_instances.loc[i, 'Node13']
  x14 = pos_instances.loc[i, 'Node14']
  x15 = pos_instances.loc[i, 'Node15']
  x17 = pos_instances.loc[i, 'Node17']
  x18 = pos_instances.loc[i, 'Node18']
  x19 = pos_instances.loc[i, 'Node19']
  x20 = pos_instances.loc[i, 'Node20']
  vw_instance = str(label) + ' | '\
  + str(x1) + ' ' + str(x2) + ' ' + str(x3) + ' ' + str(x4) + ' ' + str(x5)\
  + ' ' + str(x6) + ' ' + str(x7) + ' ' + str(x8) + ' ' + str(x9)\
  + ' ' + str(x10)+ ' ' + str(x11)+ ' ' + str(x12)+ ' ' + str(x13)\
  + ' ' + str(x14)+ ' ' + str(x15)+ ' ' + str(x17)+ ' ' + str(x18)\
  + ' ' + str(x19)+ ' ' + str(x20)
  print(vw_instance)
  vw.learn(vw_instance)


#----------------------
# Predict the treatment
# on all instances
#----------------------

appended_data = []

for j in df.index:
  x1 = df.loc[j, 'Node1']
  x2 = df.loc[j, 'Node2']
  x3 = df.loc[j, 'Node3']
  x4 = df.loc[j, 'Node4']
  x5 = df.loc[j, 'Node5']
  x6 = df.loc[j, 'Node6']
  x7 = df.loc[j, 'Node7']
  x8 = df.loc[j, 'Node8']
  x9 = df.loc[j, 'Node9']
  x10 = df.loc[j, 'Node10']
  x11 = df.loc[j, 'Node11']
  x12 = df.loc[j, 'Node12']
  x13 = df.loc[j, 'Node13']
  x14 = df.loc[j, 'Node14']
  x15 = df.loc[j, 'Node15']
  x17 = df.loc[j, 'Node17']
  x18 = df.loc[j, 'Node18']
  x19 = df.loc[j, 'Node19']
  x20 = df.loc[j, 'Node20']
  pred = vw.predict('| ' + str(x1) + ' ' + str(x2) + ' ' + str(x3)\
  + ' ' + str(x4) + ' ' + str(x5) + ' ' + str(x6) + ' ' + str(x7)\
  + ' ' + str(x8) + ' ' + str(x9) + ' ' + str(x10) + ' ' + str(x11)\
  + ' ' + str(x12) + ' ' + str(x13) + ' ' + str(x14) + ' ' + str(x15)\
  + ' ' + str(x17) + ' ' + str(x18) + ' ' + str(x19) + ' ' + str(x20))
  appended_data.append(pred)

df.loc[:, 'pred'] = appended_data

df.groupby(['pred']).size() / len(df)

df.groupby('target_control')['pred'].mean()



#------------------
# Evaluate
#------------------

df.loc[df.pred == 1].groupby(['customer_type']).size() /\
len(df.loc[df.pred == 1])


df.loc[df.pred == 2].groupby(['customer_type']).size() /\
len(df.loc[df.pred == 2])


sum(df.reward) / len(df)
# -0.755

sum(df.loc[df.target_control == df.pred, 'reward_weighted']) / len(df)

sum(df.loc[df.target_control == df.pred, 'reward']) / len(df)

