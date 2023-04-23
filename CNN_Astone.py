#!/usr/bin/env python
# coding: utf-8
'''
Red de Astone 2018

    Se propone una red 2D para clasificar entre signal y ruido
    
    Input:

        2D : mapas tiempo-frecuencia de 2s segundos por medio de discrete wavelet transforms
    
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


directory = 'data/2000 datafull_names'
dist = '0.1'

classification = 1  # 0 = binary, 1 = multiclass
batch_size = 32
trns_indx = 0 # 
fmax = 2048

X_train_filenames = np.load(directory + '/train_list_dist_' + dist + '.npy')
y_train = np.load(directory + '/train_labels_dist_' + dist + '.npy')
X_val_filenames = np.load(directory + '/val_list_dist_' + dist + '.npy')
y_val = np.load(directory + '/val_labels_dist_' + dist + '.npy')


timeInd, levels = transform_data([X_train_filenames[0]], trns_indx, fmax )[0].shape

channels = 1

filters = [8, 8, 8, 8]

dropout = 1  # 0 = soft-dropout, 1 = dropout
p = 0.3
a, b = 2, 5

inputLayer = Input((timeInd, levels, channels))

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

if not classification:
    out = Dense(units=1, activation='sigmoid')(DENSE)
else:
    out = Dense(units=2, activation='softmax')(DENSE)

model_astone = Model(inputs=[inputLayer], outputs=[out], name="model_astone")

#keras.utils.plot_model(model_astone, show_shapes=True)

model_astone.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=[
                     'accuracy', tf.keras.metrics.FalsePositives()])


my_training_batch_generator = My_Custom_Generator(
    X_train_filenames, y_train, batch_size = batch_size, trns_indx=trns_indx)
my_validation_batch_generator = My_Custom_Generator(
    X_val_filenames, y_val, batch_size = batch_size, trns_indx=trns_indx)


early_stopping = EarlyStoppingTresh(monitor='val_loss', threshold=0.0001)
history = model_astone.fit(my_training_batch_generator,
                           steps_per_epoch=int(
                               len(X_train_filenames) // batch_size),
                           validation_data=my_validation_batch_generator,
                           validation_steps=int(
                               len(X_val_filenames) // batch_size),
                           epochs=100,
                           #batch_size = 50,
                           # callbacks=[early_stopping],
                           verbose=1)


# graficar las metricas
graph_metrics(history,
              'Melspectrogram\n Metrics dist-'
              + dist
              + ' filters-'
              + str(len(filters))
              + 'val acc-'
              + str(history.history['val_accuracy'][-1]),
              'plots/mor_'+dist+'-kpc_'+str(len(filters))+'-capas.png'
              )
