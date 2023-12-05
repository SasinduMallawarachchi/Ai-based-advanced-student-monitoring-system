import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
import cv2

import pandas as pd


data = pd.read_csv('daat.csv').values


print(data[0,137])




X = data[:, 0:136].astype(float)
Y = data[:, 137]
#print(Y)
# print(dataset.shape)

# Data pre-processing
# X = dpp.head_reference(X

# Encoder the class label to number
# Converts a class vector (integers) to binary class matrix
encoder = LabelEncoder()
encoder_Y = encoder.fit_transform(Y)
matrix_Y = np_utils.to_categorical(encoder_Y)

print('metana-------------')
print(matrix_Y)
print(matrix_Y.shape)
print('metana-------------')


# Split into training and testing data
# random_state:

# train 60% val 20% test 20%
x_train, x_test, y_train, y_test = train_test_split(X, matrix_Y, test_size=0.2, random_state=120)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.4, random_state=120) # 0.25 x 0.8 = 0.2



x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1)

#####################

model = Sequential()
intput_shape = (x_train.shape[1], 1)
print(intput_shape)


model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu', input_shape=intput_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()


model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# batch_size: number of samples per gradient update
# epochs: how many times to pass through the whole training set
# verbose: show one line for every completed epoch
#history = model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=2)
history=model.fit(x_train, y_train, batch_size=1, epochs=300, verbose=1, validation_data=(x_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save('behavior_avg_class.h5')