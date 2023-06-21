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
import keras
from keras.layers import Input, Conv2D, MaxPool2D
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam

import tensorflow.compat.v2 as tf

import numpy as np

import utils_NN as ut

def main(
        
        directory = 'data/samples_names',
        dist = '10',
        batch_size = 16,
        channels = 1,
        trns_indx = 4,
        fmax = 2048,
        filters = [8, 8, 8, 8, 8, 8, 8],
        dropout = 1,  # 0 = soft-dropout, 1 = dropout
        p = 0.3,
        a = 2, b = 5
        ):    
    
   
    # choose the transform to use trns_indx
    # 0 - STFT
    # 1 - Morlet transform
    # 2 - Discrete wavelet
    # 3 - Continuous wavelet T
    # 4 - fast Continuous WT
    # 5 - Melspectrogram
    # List of names corresponding to each transformation
    trns_names = ['STFT','WT_Morlet','DWT','CWT','fCWT','MELSPECTROGRAM']
    
    
    # load the training and validation paths of data
    X_train_filenames = np.load(directory + '/train_list_dist_' + dist + '.npy')
    y_train = np.load(directory + '/train_labels_dist_' + dist + '.npy')
    X_val_filenames = np.load(directory + '/val_list_dist_' + dist + '.npy')
    y_val = np.load(directory + '/val_labels_dist_' + dist + '.npy')
    
    
    # get the shape of the transformed data
    timeInd, levels = ut.transform_data( [X_train_filenames[0]], trns_indx, fmax)[0].shape
    
    print(timeInd, levels )
    # set the filter configuration and dropout 
    
    filters_1d = [8]
    
    
    # define the input layer
    inputLayer = Input((timeInd, levels, channels))
    
    # add convolutional layers with max pooling and dropout (or soft dropout) to the model
    CONV = inputLayer
    for i in range(len(filters)):
    
        CONV = Conv2D(filters=filters[i], kernel_size=(3, 3), padding='same', activation='relu')(CONV)
        CONV = MaxPool2D(pool_size=(2, 2), strides=2)(CONV)
    
        if dropout:
            CONV = Dropout(p)(CONV)
        else:
            CONV = ut.Soft_Dropout(a, b)(CONV)
            
    for i in range(len(filters_1d)):
    
        CONV = Conv2D(filters=filters_1d[i], kernel_size=(1, 3), padding='same', activation='relu')(CONV)
        CONV = MaxPool2D(pool_size=(1, 2), strides=2)(CONV)
    
        if dropout:
            CONV = Dropout(p)(CONV)
        else:
            CONV = ut.Soft_Dropout(a, b)(CONV)
    
    DENSE = Flatten()(CONV)
    
    out = Dense(units=2, activation='softmax')(DENSE)
    
    # compile the model
    model_astone = Model(inputs=[inputLayer], outputs=[out], name="model_astone")
    model_astone.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.FalsePositives()])
    
    keras.utils.plot_model(model_astone, show_shapes=True)
    model_astone.summary()
    # define the training and validation data generators
    my_training_batch_generator = ut.My_Custom_Generator( X_train_filenames, y_train, batch_size=batch_size, trns_indx=trns_indx)
    my_validation_batch_generator = ut.My_Custom_Generator(X_val_filenames, y_val, batch_size=batch_size, trns_indx=trns_indx)
    
    
    #early_stopping = EarlyStoppingTresh(monitor='val_loss', threshold=0.0001)
    
    history = model_astone.fit(my_training_batch_generator,steps_per_epoch=int( len(X_train_filenames) // batch_size),
                               validation_data=my_validation_batch_generator,
                               validation_steps=int( len(X_val_filenames) // batch_size),
                               epochs=100,
                               verbose=1)
    
    path = "redes/"+trns_names[trns_indx]+"_dist-" + dist + ".h5"
    model_astone.save(path)
    
    # graph metrics
    ut.graph_metrics(history,
                  trns_names[trns_indx]
                  +'\n Metrics dist-'
                  + dist
                  + ' filters-'
                  + str(len(filters))
                  + 'val acc-'
                  + str(history.history['val_accuracy'][-1]),
                  'plots/'+trns_names[trns_indx]+'_'+dist+'-kpc.png'
                  )
