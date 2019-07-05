# https://arxiv.org/abs/1807.09809

import pandas as pd
import numpy as np
from pathlib import Path
from keras import models
from keras import layers


home = str(Path.home())

input_file = home + '/vehicle-dataset/dat_as_cb_loss_only.csv'



######################
# Load data
######################

df = pd.read_csv(input_file)


df.shape
# (846, 23)



x_cols = [col for col in df if col.startswith('X')]

df['Y']


df_x = df.loc[:, x_cols]
df_x.shape
# (846, 18)



######################
# Define model
######################

model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(18,)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(np.asarray(df_x),
                    np.asarray(df['Y']),
                    epochs=20,
                    batch_size=16)
