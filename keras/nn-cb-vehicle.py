# https://arxiv.org/abs/1807.09809

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


home = str(Path.home())

input_file = home + '/vehicle-dataset/dat_as_cb_loss_only.csv'



######################
# Load data
######################

df = pd.read_csv(input_file)

df.shape
# (846, 23)



######################
# Prepare data
######################

x_cols = [col for col in df if col.startswith('X')]

df_x = df.loc[:, x_cols]
df_x.shape
# (846, 18)

nn_input_x = np.asarray(df_x)




df['Y_code'] = None
df['Y_code'].loc[df.Y == 'opel'] = 0
df['Y_code'].loc[df.Y == 'saab'] = 1
df['Y_code'].loc[df.Y == 'bus'] = 2
df['Y_code'].loc[df.Y == 'van'] = 3

nn_input_y = to_categorical(df['Y_code'])





######################
# Split data
######################

nn_input_x_train, nn_input_x_test, nn_input_y_train, nn_input_y_test = train_test_split(nn_input_x, nn_input_y, test_size=0.2, random_state=666)

[x.shape for x in [nn_input_x_train, nn_input_x_test, nn_input_y_train, nn_input_y_test]]
# [(676, 18), (170, 18), (676, 4), (170, 4)]



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




######################
# Train model
######################

history = model.fit(nn_input_x_train,
                    nn_input_y_train,
                    epochs=20,
                    batch_size=16)



######################
# Plot results
######################

plt.clf()
loss = history.history['loss']
# val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


######################
# Predict test data
######################

model.predict(nn_input_x_test)
