# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 02:36:13 2023

Cambios a propuesta 4, red generada automaticamente

@author: benja
"""

# importing libraries
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPool2D, Concatenate,Conv1D
from keras.models import Model
import keras
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam
import tensorflow.compat.v2 as tf

import numpy as np

import utils_NN as ut


def main(directory = 'data/samples_names',
    dist = '10'):
    # set the directory, distance, and batch size
    # directory = '/media/guillermo/Swap/Data Supernovae/samples_names'
    
    trns_indx = 1 
    batch_size = 32
    channels = 1
    trns_cnt = 3
    fmax = 2048
    
    # load the training and validation paths of data
    X_train_filenames = np.load(directory + '/train_list_dist_' + dist + '.npy')
    y_train = np.load(directory + '/train_labels_dist_' + dist + '.npy')
    X_val_filenames = np.load(directory + '/val_list_dist_' + dist + '.npy')
    y_val = np.load(directory + '/val_labels_dist_' + dist + '.npy')
    
    # get the shape of the transformed data
    
    dim1_list, dim2_list = [], []
    for i in range(trns_cnt):
        dim1, dim2 = ut.transform_data_stft_3( [X_train_filenames[0]], fmax, trns_indx = trns_indx)[i][0].shape
        dim1_list.append(dim1)
        dim2_list.append(dim2)

    p = 0.3
    
    inputs = []
    toconcat = []
    
    for i in range(trns_cnt):
        ker = (3, 3)
        pool_size=(2, 2)
        
        inputLayer = Input((dim1_list[i], dim2_list[i], channels))
        CONV = inputLayer
        
        dim_menor = dim2_list[i]
        dim_mayor = dim1_list[i]
        
        if dim1_list[i] < dim2_list[i]:
            dim_menor = dim1_list[i]
            dim_mayor = dim2_list[i]
            
        while dim_menor > 1 or dim_mayor >= 8 :
            
            if dim_menor == 1 and dim_mayor >= 8 :
                ker = (1, 3)
                pool_size=(1, 2)
            
            CONV = Conv2D(filters=8, kernel_size=ker, padding='same', activation='relu')(CONV)
            CONV = MaxPool2D(pool_size=pool_size, strides=2)(CONV)
            CONV = Dropout(p)(CONV)
            dim_menor//=2
            dim_mayor//=2
            
        DENSE = Flatten()(CONV)
        inputs.append(inputLayer)
        toconcat.append(DENSE)
    
    # concatenate the flattened outputs of convolutional layers
    DENSE = Concatenate()(toconcat)
    
    # add dense layer with 64 units and dropout layer
    DENSE = Dense(units= 64)(DENSE)
    
    # add output layer with softmax activation
    out = Dense(units=2, activation='softmax')(DENSE)
       
    # define model
    model_pps = Model(inputs=inputs,outputs=[out], name = "model_pps")
    
    
       
    model_pps.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.FalsePositives()])
    # model_pps.summary()
    # keras.utils.plot_model(model_pps, show_shapes=True)
    
    # define the training and validation data generators
    my_training_batch_generator = ut.My_Custom_Generator( X_train_filenames, y_train, batch_size=batch_size, comp=True,trns_indx=trns_indx)
    my_validation_batch_generator = ut.My_Custom_Generator(X_val_filenames, y_val, batch_size=batch_size, comp=True,trns_indx=trns_indx)
    
    
    history = model_pps.fit(my_training_batch_generator,steps_per_epoch=int( len(X_train_filenames) // batch_size),
                                validation_data=my_validation_batch_generator,
                                validation_steps=int( len(X_val_filenames) // batch_size),
                                epochs=100,
                                verbose=1)
    
    path = "redes/proposal6_dist-" + dist + ".h5"
    model_pps.save(path)
    
    # graph metrics
    ut.graph_metrics(history,
                  'proposal\n Metrics dist-'
                  + dist
                  + 'val acc-'
                  + str(history.history['val_accuracy'][-1]),
                  'plots/proposal6_'+dist+'-kpc.png'
                  )
    
main()