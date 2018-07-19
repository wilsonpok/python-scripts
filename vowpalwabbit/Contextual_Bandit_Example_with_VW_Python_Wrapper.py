# load required packages
import pandas as pd
import sklearn as sk
import numpy as np

# import vowpal wabbit's python wrapper
from vowpalwabbit import pyvw


# # install vowpal wabbit
# !pip install boost
# !apt-get install libboost-program-options-dev zlib1g-dev libboost-python-dev -y
# !pip install vowpalwabbit


# generate sample data that could originate from previous random trial, e.g. AB test, for the CB to explore
## data here are equivalent to example in https://github.com/JohnLangford/vowpal_wabbit/wiki/Contextual-Bandit-Example
train_data = [{'action': 1, 'cost': 2, 'probability': 0.4, 'feature1': 'a', 'feature2': 'c', 'feature3': ''},
              {'action': 3, 'cost': 0, 'probability': 0.2, 'feature1': 'b', 'feature2': 'd', 'feature3': ''},
              {'action': 4, 'cost': 1, 'probability': 0.5, 'feature1': 'a', 'feature2': 'b', 'feature3': ''},
              {'action': 2, 'cost': 1, 'probability': 0.3, 'feature1': 'a', 'feature2': 'b', 'feature3': 'c'},
              {'action': 3, 'cost': 1, 'probability': 0.7, 'feature1': 'a', 'feature2': 'd', 'feature3': ''}]

train_df = pd.DataFrame(train_data)

## add index to df
train_df['index'] = range(1, len(train_df) + 1)
train_df = train_df.set_index("index")

# generate some test data that you want the CB to make decisions for, e.g. features describing new users, for the CB to exploit
test_data = [{'feature1': 'b', 'feature2': 'c', 'feature3': ''},
            {'feature1': 'a', 'feature2': '', 'feature3': 'b'},
            {'feature1': 'b', 'feature2': 'b', 'feature3': ''},
            {'feature1': 'a', 'feature2': '', 'feature3': 'b'}]

test_df = pd.DataFrame(test_data)

## add index to df
test_df['index'] = range(1, len(test_df) + 1)
test_df = test_df.set_index("index")

# take a look at dataframes
print(train_df)
print(test_df)


# create python model - this stores the model parameters in the python vw object; here a contextual bandit with four possible actions
vw = pyvw.vw("--cb 4")

# use the learn method to train the vw model, train model row by row using a loop
# i==1
for i in train_df.index:
  ## provide data to cb in requested format
  action = train_df.loc[i, "action"]
  cost = train_df.loc[i, "cost"]
  probability = train_df.loc[i, "probability"]
  feature1 = train_df.loc[i, "feature1"]
  feature2 = train_df.loc[i, "feature2"]
  feature3 = train_df.loc[i, "feature3"]
  ## do the actual learning
  vw.learn(str(action)+":"+str(cost)+":"+str(probability)+" | "+str(feature1)+" "+str(feature2)+" "+str(feature3))

# use the same model object that was trained to perform predictions

# predict row by row and output results
# j==1
for j in test_df.index:
  feature1 = test_df.loc[j, "feature1"]
  feature2 = test_df.loc[j, "feature2"]
  feature3 = test_df.loc[j, "feature3"]
  choice = vw.predict("| "+str(feature1)+" "+str(feature2)+" "+str(feature3))
  print(j, choice)

# the CB assigns every instance to action 3 as it should per the cost structure of the train data; you can play with the cost structure to see that the CB updates its predictions accordingly

# BONUS: save and load the CB model
# save model
vw.save('cb.model')
del vw
# load from saved file
vw = pyvw.vw("--cb 4 -i cb.model")
print(vw.predict('| a b'))
