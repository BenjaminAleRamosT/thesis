#!/usr/bin/env python
# coding: utf-8


import sys
import os
import numpy as np
import Preproceso as pr
import keras
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import glob
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.utils import control_flow_util
from keras.engine import base_layer
from keras import backend
import torch
import tensorflow.compat.v2 as tf
import numbers
from scipy.stats import beta, bernoulli


def beta_drop(inputs, a, b):
    """
    Applies BetaDropout regularization to the input tensor.

    :param inputs: Input tensor.
    :param a: First shape parameter of the Beta distribution.
    :param b: Second shape parameter of the Beta distribution.

    :return: Regularized tensor.
    """
    
    x = tf.convert_to_tensor(inputs, name="x")
    x_dtype = x.dtype

    # Check if a and b are real numbers.
    is_a_number = isinstance(a, numbers.Real)
    is_b_number = isinstance(b, numbers.Real)
    
    # If a and b are not tensors, then they should be scalar values.
    # Calculate the scaling factor and scale the input tensor by the scaling factor.
    if not tf.is_tensor(a) or not tf.is_tensor(b):
        if is_a_number and is_b_number:
            keep_prob = 1 - a/(a+b)
            scale = 1 / keep_prob
            scale = tf.convert_to_tensor(scale, dtype=x_dtype)
            ret = tf.math.multiply(x, scale)
        else:
            raise ValueError(
                f"`a` and 'b' must be a scalar or scalar tensor. Received: a={a}, b={b}")
    else:
        # Check if a and b are scalar tensors.
        a.get_shape().assert_has_rank(0)
        b.get_shape().assert_has_rank(0)

        a_dtype = a.dtype
        b_dtype = b.dtype

        # If the data type of a and b are not compatible with the data type of the input tensor,
        # then cast a and b to the data type of the input tensor.
        if a_dtype != x_dtype or b_dtype != x_dtype:
            if not a_dtype.is_compatible_with(x_dtype) or not b_dtype.is_compatible_with(x_dtype):
                raise ValueError(
                    "`x.dtype` must be compatible with `a.dtype` and `b.dtype`. "
                    f"Received: x.dtype={x_dtype} and a.dtype={a_dtype}, b.dtype={b_dtype}")
            a = tf.cast(a, x_dtype, name="a")
            b = tf.cast(b, x_dtype, name="b")
        one_tensor = tf.constant(1, dtype=x_dtype)
        ret = tf.realdiv(x, tf.math.subtract(one_tensor, a/(a+b)))

    size = x.shape[1:]
    
    # Sample random values from the Beta distribution.
    random_tensor = beta.rvs(a, b, size=size)
    random_tensor = tf.convert_to_tensor(random_tensor, dtype=x_dtype)
    
    # Scale the output by multiplying it with the random values sampled from the Beta distribution.
    ret = tf.math.multiply(ret, random_tensor)

    return ret


# capa soft-dropout


class Soft_Dropout(base_layer.BaseRandomLayer):
    """Applies Dropout to the input.
    Args:
      a: Float higher than 0.
      b: Float higher than 0.
      noise_shape: 1D integer tensor representing the shape of the
        binary dropout mask that will be multiplied with the input.
        For instance, if your inputs have shape
        `(batch_size, timesteps, features)` and
        you want the dropout mask to be the same for all timesteps,
        you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, a, b, noise_shape=None, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if isinstance(a, (int, float)) and not 0 < a:  # cambiar para condiciones de a y b > 0
            raise ValueError(
                f"Invalid value {a} received for "
                "`a`, expected a value higher than 0."
            )
        if isinstance(b, (int, float)) and not 0 < b:  # cambiar para condiciones de a y b > 0
            raise ValueError(
                f"Invalid value {b} received for "
                "`b`, expected a value higher than 0."
            )
        self.a = a
        self.b = b
        self.scale = a/(a+b)
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        # Subclasses of `Dropout` may implement `_get_noise_shape(self,
        # inputs)`, which will override `self.noise_shape`, and allows for
        # custom noise shapes with dynamically sized inputs.
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = tf.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(
                concrete_inputs_shape[i] if value is None else value
            )
        return tf.convert_to_tensor(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        def dropped_inputs():

            return beta_drop(
                inputs, self.a, self.b
            )

        output = control_flow_util.smart_cond(
            # lambda: tf.multiply(inputs, self.scale) #
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "a": self.a,
            "b": self.b,
            "scale": self.scale,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EarlyStoppingTresh(tf.keras.callbacks.Callback):
    """
   EarlyStoppingTresh Callback to stop the execution when the monitored metric
   reaches the threshold value.

   :param monitor: metric to monitor
   :param threshold: threshold value, training will be stopped if the monitored
                     metric is less than or equal to this value.
   """

    def __init__(self, monitor='loss', threshold=0.1):
        super(EarlyStoppingTresh, self).__init__()
        self.threshold = threshold
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called at the end of each epoch.

        :param epoch: current epoch number
        :param logs: dictionary containing the metrics results for this epoch
        """
        metric = logs[self.monitor]
        if metric <= self.threshold:
            print('\nEpoch %d: Reached threshold, terminating training' % (epoch))
            self.model.stop_training = True


def load_data(directory, dist, sep=' ', multclass=False):
    """
    load_data Carga toda la data de una distancia en el directorio especificado

    :param directory: directorio donde se encuentran los archivos
    :param dist: distancia de las simulaciones ['0.1', '3.23', '6.87', '10']
    :param sep: separador que usan los archivos [' ', ',']
    :param multclass: define si los labels son categoricos o binarios
    :return: dos variables, x la data : y los labels de cada dato
    """

    signal = []
    #files = glob.glob(directory+'/s11.2--LS220_'+dist+'kpc_sim*.txt')

    files = glob.glob(directory+'/s11.2--LS220_'+dist+'kpc_sim*.txt')

    for file in files:
        # Use the pandas.read_csv() function to read the contents of the file
        data = pd.read_csv(file, sep=sep, header=None)
        data = data.values.T
        # print(data.shape)
        # Add the contents of the file to the numpy array
        signal.append(data)

    noise = []
    files = glob.glob(directory+'/aLIGO_noise_2sec_sim*.txt')
    for file in files:
        # Use the pandas.read_csv() function to read the contents of the file
        data = pd.read_csv(file, sep=sep, header=None)
        # print(data.shape)
        data = data.values.T

        # Add the contents of the file to the numpy array
        noise.append(data)

    if len(noise) != len(signal):
        if len(noise) > len(signal):
            noise, n = train_test_split(noise, train_size=len(signal))
        else:
            signal, n = train_test_split(signal, train_size=len(noise))

    y1 = np.zeros(len(noise))
    y2 = np.ones(len(signal))

    x = np.vstack((signal, noise))

    y = np.hstack((y1, y2))

    if multclass:
        y = to_categorical(y, dtype="uint8")

    if x.dtype == 'O':
        x = x.astype(complex)
        x = abs(x)

    return x, y



def norm_data(x):
    """
    Normalizes the data in x
    :param x: input data to be normalized
    :return: normalized x
    """
    dfmax, dfmin = np.max(x), np.min(x)
    x = (x - dfmin)/(dfmax - dfmin)

    return x


def graph_metrics(history, name, filename):
    """
    Plots the metrics: loss, val_loss, accuracy, and val_accuracy
    :param history: training history dictionary
    :param name: plot title
    :param filename: name of the file where the plot will be saved
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    fig1, ax1 = plt.subplots()

    ax1.plot(epochs, loss, 'y', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')

    ax1.plot(epochs, acc, 'b', label='Acc')
    ax1.plot(epochs, val_acc, 'purple', label='val_acc')

    ax1.set_title(name)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    fig1.savefig(filename)
    plt.close(fig1)


def name_store(directory='data/samples'):
    """
    Creates train and validation sets for signal and noise files and stores them in a directory
    :param directory: directory containing the signal and noise files
    """
    
    # Define a list of distances
    dist = ['0.1','2.71', '5.05','7.39', '10']
    
    # model = ['LS220', 'GShen', 'SFHo']
    
    # Loop through each distance value
    for i in range(len(dist)):
        # Find all signal files for the current distance value
        # signal_names = glob.glob(directory +'/s*--LS220_'+ dist[i] +'kpc_sim*.txt')
        signal_names = glob.glob(directory +'/s*--*_'+ dist[i] +'kpc_sim*.txt')

        
        # Find all noise files, this could be done once
        noise_names = glob.glob(directory+'/aLIGO_noise_2sec_sim*.txt')

        # If the number of noise files is not equal to the number of signal files,
        # split the larger set to match the size of the smaller set
        if len(noise_names) != len(signal_names):
            if len(noise_names) > len(signal_names):
                noise_names, n = train_test_split(
                    noise_names, train_size=len(signal_names))
            else:
                signal_names, n = train_test_split(
                    signal_names, train_size=len(noise_names))

        # Create arrays for the labels of the signal and noise files
        y1 = np.zeros(len(signal_names))
        y2 = np.ones(len(noise_names))

        # Concatenate the arrays of signal and noise file names
        x_names = signal_names + noise_names

        # Convert labels to categorical format
        y = np.hstack((y1,y2))
        y = to_categorical(y, dtype="uint8")

        # Shuffle the file names and labels
        x_shuffled, y_shuffled = shuffle(x_names, y)

        # Create a new directory to store the file names and labels
        outdir = directory + '_names'
        os.makedirs(outdir, exist_ok=True)
        
        # Split the shuffled file names and labels into training and validation sets
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(x_shuffled, y_shuffled, test_size=0.3, random_state=1)

        
        # Save the training set file names and labels to disk
        name_x_t = outdir + '/train_list_dist_' + dist[i] + '.npy'
        np.save(name_x_t, X_train_filenames)

        name_x_v = outdir + '/val_list_dist_' + dist[i] + '.npy'
        np.save(name_x_v, X_val_filenames)

        # Save the validation set file names and labels to disk
        name_y_t = outdir + '/train_labels_dist_' + dist[i] + '.npy'
        np.save(name_y_t, y_train)

        name_y_v = outdir + '/val_labels_dist_' + dist[i] + '.npy'
        np.save(name_y_v, y_val)

# name_store()
def transform_data(batch_x , trns_indx, fmax = 2048):
    """
    Apply a specified transform function to a batch of data files
    :param batch_x: list of file names
    :param trns_indx: index of the transform function to apply
    :param fmax: maximum frequency to consider
    :return: transformed data
    """
    
    # Define a list of transform functions
    trns_funct = [pr.calcular_stft, 
                  pr.wt_morlet,
                  pr.dwt_pywt, 
                  pr.calcular_cwt, 
                  pr.calcular_fcwt, 
                  pr.mel]
    
    # Apply the specified transform function to each file in the batch
    
    x_transform = [np.loadtxt(
        fname = str(file_name), delimiter=' ', usecols=1)
         for file_name in batch_x]
    
    x = [
        abs(trns_funct[trns_indx](
            t,
            graph = False, 
            fmax = fmax )
            ) 
        if  t.shape == (16384,) 
        else 
        abs(trns_funct[trns_indx](
            t[::2],
            graph = False, 
            fmax = fmax )
            ) 
        
        for t in x_transform]
        
    x = norm_data(np.array(x))
    return x

def transform_data_stft_3(batch_x , fmax = 2048, trns_indx = 0):
    """
    Apply the STFT transformation function three times to a batch of data files at different resolutions.    
    :param batch_x: list of file names
    :param trns_indx: index of the transform function to apply
    :param fmax: maximum frequency to consider
    :return: transformed data
    """
    
    x_transform = [np.loadtxt(
        fname = str(file_name), delimiter=' ', usecols=1)
         for file_name in batch_x]
    
    
    
    
    trns_funct = [pr.calcular_stft,
                  pr.mel]
    
    windL = [512, 1024, 2048]
    
    n_fft = [512, 1024, 2048]
    hop_length = [32, 64, 256]
    n_mels = [20, 40, 60]
    
    x= []
    # Apply the specified transform function to each file in the batch
    if trns_indx == 0:
        for i in range(len(windL)):
            x_a = [abs(trns_funct[trns_indx](
                t ,graph = False, fmax = fmax, window_length = windL[i]))
                if  t.shape == (16384,) 
                else
                abs(trns_funct[trns_indx](
                    t[::2] ,graph = False, fmax = fmax, window_length = windL[i]))
                for t in x_transform]
            x_a = norm_data(np.array(x_a))
            x.append(x_a)
    else :
        for i in range(len(n_fft)):
            x_a = [abs(trns_funct[trns_indx](
                t ,graph = False, fmax = fmax,n_fft = n_fft[i], hop_length = hop_length[i], n_mels = n_mels[i]))
                if  t.shape == (16384,) 
                else
                abs(trns_funct[trns_indx](
                    t[::2] ,graph = False, fmax = fmax,n_fft = n_fft[i], hop_length = hop_length[i], n_mels = n_mels[i]))
                for t in x_transform]
            x_a = norm_data(np.array(x_a))
        
            x.append(x_a)
    
    
    return x



class My_Custom_Generator(keras.utils.Sequence):
    """
    Custom generator class for feeding data into a Keras model during training or evaluation
    """
    def __init__(self, image_filenames, labels, batch_size = 32, trns_indx = 0,fmax = 2048, comp=False):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.trns_indx = trns_indx
        self.fmax = fmax
        self.comp = comp

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """
        Get a batch of data
        :param idx: index of the batch
        :return: tuple containing transformed data and corresponding labels
        """
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        if self.comp:
            x = transform_data_stft_3(batch_x , self.fmax)
        else:
            x = transform_data(batch_x , self.trns_indx, self.fmax)
    
        return x , np.array(batch_y)


