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

# agg.iloc[agg.index.get_level_values('target_control') == 'target']
# agg.iloc[agg.index.get_level_values('outcome') == 'positive']


#------------------
# Process data
#------------------

df.loc[df.outcome == 'negative', 'outcome'] = 3
df.loc[df.outcome == 'positive', 'outcome'] = 0

df.loc[df.target_control == 'target', 'prob_target'] = 0.5063
df.loc[df.target_control == 'control', 'prob_target'] = 1 - 0.5063

df.loc[df.target_control == 'target', 'target_control'] = 1
df.loc[df.target_control == 'control', 'target_control'] = 2



#------------------
# Train model
#------------------

vw = pyvw.vw('--cb 2')

for i in df.index:
  action = df.loc[i, 'target_control']
  cost = df.loc[i, 'outcome']
  probability = df.loc[i, 'prob_target']
  x1 = df.loc[i, 'Node1']
  x2 = df.loc[i, 'Node2']
  x3 = df.loc[i, 'Node3']
  x4 = df.loc[i, 'Node4']
  x5 = df.loc[i, 'Node5']
  x6 = df.loc[i, 'Node6']
  x7 = df.loc[i, 'Node7']
  x8 = df.loc[i, 'Node8']
  x9 = df.loc[i, 'Node9']
  x10 = df.loc[i, 'Node10']
  x11 = df.loc[i, 'Node11']
  x12 = df.loc[i, 'Node12']
  x13 = df.loc[i, 'Node13']
  x14 = df.loc[i, 'Node14']
  x15 = df.loc[i, 'Node15']
  x17 = df.loc[i, 'Node17']
  x18 = df.loc[i, 'Node18']
  x19 = df.loc[i, 'Node19']
  x20 = df.loc[i, 'Node20']
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
# Predict
#------------------

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
  choice = vw.predict('| ' + str(x1) + ' ' + str(x2) + ' ' + str(x3)\
  + ' ' + str(x4) + ' ' + str(x5) + ' ' + str(x6) + ' ' + str(x7)\
  + ' ' + str(x8) + ' ' + str(x9) + ' ' + str(x10) + ' ' + str(x11)\
  + ' ' + str(x12) + ' ' + str(x13) + ' ' + str(x14) + ' ' + str(x15)\
  + ' ' + str(x17) + ' ' + str(x18) + ' ' + str(x19) + ' ' + str(x20))
  appended_data.append(choice)

df.loc[:, 'recommendation'] = appended_data

df.groupby(['recommendation']).size() / len(df)

df.loc[df.recommendation == 1].groupby(['customer_type']).size() / len(df.loc[df.recommendation == 1])
# lost_cause      0.279412
# persuadable     0.328431
# sleeping_dog    0.215686
# sure_thing      0.176471
