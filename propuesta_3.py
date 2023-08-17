# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 02:36:13 2023

@author: benja
"""

# importing libraries
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPool2D, Concatenate, Conv1D, MaxPool1D
from keras.models import Model
import keras
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam
import tensorflow.compat.v2 as tf

import numpy as np

import utils_NN as ut



# set the directory, distance, and batch size
directory = 'data/2000 datafull_names'
dist = '10'
batch_size = 32
channels = 1

fmax = 2048

# load the training and validation paths of data
X_train_filenames = np.load(directory + '/train_list_dist_' + dist + '.npy')
y_train = np.load(directory + '/train_labels_dist_' + dist + '.npy')
X_val_filenames = np.load(directory + '/val_list_dist_' + dist + '.npy')
y_val = np.load(directory + '/val_labels_dist_' + dist + '.npy')

# get the shape of the transformed data


timeInd_a, levels_a = ut.transform_data_stft_3( [X_train_filenames[0]], fmax)[0][0].shape
timeInd_b, levels_b = ut.transform_data_stft_3( [X_train_filenames[0]], fmax)[1][0].shape
timeInd_c, levels_c = ut.transform_data_stft_3( [X_train_filenames[0]], fmax)[2][0].shape

# set the filter configuration and dropout 
blocks1 = [8, 8, 8, 16, 16, 32]
blocks2 = [8, 8, 8, 8, 16]
blocks2_1D = [16, 32]
blocks3 = [8, 8, 8, 8]
blocks3_1D = [16, 16, 16, 32]
dropout = 1  # 0 = soft-dropout, 1 = dropout
p = 0.3
a, b = 2, 5


#set input layers
inputLayer_a = Input((timeInd_a, levels_a, channels))
inputLayer_b = Input((timeInd_b, levels_b, channels))
inputLayer_c = Input((timeInd_c, levels_c, channels))

inputs = []
toconcat = []



# add convolutional layers with max pooling and dropout (or soft dropout) to the model first line
CONV_a = inputLayer_a
for i in range(len(blocks1)):

    CONV_a = Conv2D(filters=blocks1[i], kernel_size=(3, 3), padding='same', activation='relu')(CONV_a)
    CONV_a = MaxPool2D(pool_size=(2, 2), strides=2)(CONV_a)

    if dropout:
        CONV_a = Dropout(p)(CONV_a)
    else:
        CONV_a = ut.Soft_Dropout(a, b)(CONV_a)

DENSE_a = Flatten()(CONV_a)

# add convolutional layers with max pooling and dropout (or soft dropout) to the model second line
CONV_b = inputLayer_b
for i in range(len(blocks2)):

    CONV_b = Conv2D(filters=blocks2[i], kernel_size=(3, 3), padding='same', activation='relu')(CONV_b)
    CONV_b = MaxPool2D(pool_size=(2, 2), strides=2)(CONV_b)

    if dropout:
        CONV_b = Dropout(p)(CONV_b)
    else:
        CONV_b = ut.Soft_Dropout(a, b)(CONV_b)
        
for i in range(len(blocks2_1D)):

    CONV_b = Conv1D(filters=blocks2_1D[i], kernel_size=(3), padding='same', activation='relu')(CONV_b)
    CONV_b = MaxPool2D(pool_size=(2,1), strides=2)(CONV_b)

    if dropout:
        CONV_b = Dropout(p)(CONV_b)
    else:
        CONV_b = ut.Soft_Dropout(a, b)(CONV_b)


DENSE_b = Flatten()(CONV_b)

# add convolutional layers with max pooling and dropout (or soft dropout) to the model third line
CONV_c = inputLayer_c
for i in range(len(blocks3)):

    CONV_c = Conv2D(filters=blocks3[i], kernel_size=(3, 3), padding='same', activation='relu')(CONV_c)
    CONV_c = MaxPool2D(pool_size=(2, 2), strides=2)(CONV_c)

    if dropout:
        CONV_c = Dropout(p)(CONV_c)
    else:
        CONV_c = ut.Soft_Dropout(a, b)(CONV_c)
        
for i in range(len(blocks3_1D)):

    CONV_c = Conv1D(filters=blocks3_1D[i], kernel_size=(3), padding='same', activation='relu')(CONV_c)
    CONV_c = MaxPool2D(pool_size=(2,1), strides=2)(CONV_c)

    if dropout:
        CONV_c = Dropout(p)(CONV_c)
    else:
        CONV_c = ut.Soft_Dropout(a, b)(CONV_c)



DENSE_c = Flatten()(CONV_c)

# append input layers to input list
inputs.append(inputLayer_a)
inputs.append(inputLayer_b)
inputs.append(inputLayer_c)

# append flattened outputs of convolutional layers to output list
toconcat.append(DENSE_a)
toconcat.append(DENSE_b)
toconcat.append(DENSE_c)

# concatenate the flattened outputs of convolutional layers
DENSE = Concatenate()(toconcat)

# add dense layer with 64 units and dropout layer

#DENSE = Dense(units= 64)(DENSE)
#DENSE = Dropout(p)(DENSE)


# add output layer with softmax activation
out = Dense(units=2, activation='softmax')(DENSE)
   
#define model
model_pps3 = Model(inputs=inputs,outputs=[out], name = "model_pps3")


   
model_pps3.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.FalsePositives()])
# model_pps3.summary()
#keras.utils.plot_model(model_pps3, show_shapes=True)

# define the training and validation data generators
my_training_batch_generator = ut.My_Custom_Generator( X_train_filenames, y_train, batch_size=batch_size, comp=True)
my_validation_batch_generator = ut.My_Custom_Generator(X_val_filenames, y_val, batch_size=batch_size, comp=True)


history = model_pps3.fit(my_training_batch_generator,steps_per_epoch=int( len(X_train_filenames) // batch_size),
                            validation_data=my_validation_batch_generator,
                            validation_steps=int( len(X_val_filenames) // batch_size),
                            epochs=100,
                            verbose=1)

path = "redes/proposal3_dist-" + dist+".h5"
model_pps3.save(path)

# graph metrics
ut.graph_metrics(history,
              'proposal_3\n Metrics dist-'
              + dist
              + 'val acc-'
              + str(history.history['val_accuracy'][-1]),
              'plots/proposal3_'+dist+'-kpc.png'
              )