#!/usr/bin/env python
# coding: utf-8
'''
Astone Network 2018
A 2D network is proposed to classify between signal and noise.

Input:
2D: Time-frequency maps of 2 seconds using discrete wavelet transforms.
'''


# importing libraries
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Concatenate
from keras.models import Model
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam, SGD

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow.compat.v2 as tf

import numpy as np

from utils_NN import*

# set the directory, distance, and batch size
directory = 'data/2000 datafull_names'
dist = '0.1'
batch_size = 32
channels = 1

# choose the transform to use trns_indx
# 0 - STFT
# 1 - Morlet transform
# 2 - Discrete wavelet
# 3 - Continuous wavelet T
# 4 - fast Continuous WT
# 5 - Melspectrogram

trns_indx = 0
fmax = 2048

# load the training and validation paths of data
X_train_filenames = np.load(directory + '/train_list_dist_' + dist + '.npy')
y_train = np.load(directory + '/train_labels_dist_' + dist + '.npy')
X_val_filenames = np.load(directory + '/val_list_dist_' + dist + '.npy')
y_val = np.load(directory + '/val_labels_dist_' + dist + '.npy')


# get the shape of the transformed data
timeInd, levels = transform_data( [X_train_filenames[0]], trns_indx, fmax)[0].shape


# set the filter configuration and dropout 
filters = [8, 8, 8, 8]
dropout = 1  # 0 = soft-dropout, 1 = dropout
p = 0.3
a, b = 2, 5

# define the input layer
inputLayer = Input((timeInd, levels, channels))

# add convolutional layers with max pooling and dropout (or soft dropout) to the model
CONV = inputLayer
for i in range(len(filters)):

    CONV = Conv2D(filters=filters[i], kernel_size=(
        3, 3), padding='same', activation='relu')(CONV)
    CONV = MaxPool2D(pool_size=(2, 2), strides=2)(CONV)

    if dropout:
        CONV = Dropout(p)(CONV)
    else:
        CONV = Soft_Dropout(a, b)(CONV)

DENSE = Flatten()(CONV)

out = Dense(units=2, activation='softmax')(DENSE)

# compile the model
model_astone = Model(inputs=[inputLayer], outputs=[out], name="model_astone")
model_astone.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.FalsePositives()])

# define the training and validation data generators
my_training_batch_generator = My_Custom_Generator( X_train_filenames, y_train, batch_size=batch_size, trns_indx=trns_indx)
my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size=batch_size, trns_indx=trns_indx)


#early_stopping = EarlyStoppingTresh(monitor='val_loss', threshold=0.0001)

history = model_astone.fit(my_training_batch_generator,steps_per_epoch=int( len(X_train_filenames) // batch_size),
                           validation_data=my_validation_batch_generator,
                           validation_steps=int( len(X_val_filenames) // batch_size),
                           epochs=100,
                           verbose=1)

# graph metrics
graph_metrics(history,
              'Melspectrogram\n Metrics dist-'
              + dist
              + ' filters-'
              + str(len(filters))
              + 'val acc-'
              + str(history.history['val_accuracy'][-1]),
              'plots/mor_'+dist+'-kpc_'+str(len(filters))+'-capas.png'
              )
