import numpy as np
import matplotlib.pyplot as plt

import pywt
import os
from pathlib import Path

import mne




def shift(directory,outdir):
    """
    Shifts the data in the input directory by 4 different positions
    and saves each shifted file in the output directory with the
    original filename and an index indicating the shift position.

    Parameters:
    -----------
    directory : str
        Path to the directory containing the input files.
    outdir : str
        Path to the directory where the output files will be saved.


    """
    
    # create the output directory if it does not exist
    os.makedirs(outdir, exist_ok=True)
    
    # loop over all files in the input directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        
        # checking if it is a file
        if os.path.isfile(f):
            
            # load the data from the input file
            data = np.loadtxt(f, delimiter=" ",)
            
            # shift the data 4 positions to the left and save each
            # shifted file to the output directory with a different index
            for i in range(4):
                shift = ( i ) * 4915
                data_shifted = np.roll(data, shift,axis=0)
                
                # create the output file path and save the shifted data
                outfile =  outdir + '/' + Path(filename).stem + '_' + str(i) + '.txt'
                np.savetxt(outfile, data_shifted, delimiter=' ')
        

import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

def calcular_stft(data,sampling_rate = 16384 ,graph=False, fmax=8192,  window_length = 2048, num_segments = 144, step_ratio=4):
    """
    Calculate the Short-Time Fourier Transform (STFT) of a signal.

    Parameters:
    -----------
    data: array_like
        The signal for which to calculate the STFT.
    sampling_rate: int
        The sampling rate of the signal.
    graph: bool
        Whether to plot the spectrogram of the STFT.
    fmax: int
        The maximum frequency to plot in the spectrogram.

    Returns:
    --------
    Zxx: array_like
        The complex STFT coefficients.
    """

    # Define STFT parameters
    window_step = window_length // step_ratio
    freq_limit = fmax
    
    # Calculate STFT using the scipy.signal.stft function
    f, t, Zxx = stft(data, fs=sampling_rate, window='hann', nperseg=window_length, noverlap=window_step, nfft=window_length)

    # Slice the Zxx matrix to the maximum frequency of interest
    freq_range = (0, freq_limit)  # Hz
    freq_slice = np.where((f >= freq_range[0]) & (f <= freq_range[1]))

    # keep only frequencies of interest
    f = f[freq_slice]
    Zxx = Zxx[freq_slice, :][0]

    if graph:
        # Plot the spectrogram
        plt.pcolormesh(t, f, np.abs(Zxx), cmap='twilight', vmax=abs(Zxx).max(), vmin=-abs(Zxx).max())
        plt.title('STFT\nfreq_lim=' + str(freq_limit))
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.show()

    return Zxx


def wt_morlet(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    """
    Compute the Wavelet Transform (WT) using the Morlet wavelet for a given input signal.

    Parameters:
        data (array): The input signal to be transformed.
        sampling_rate (float): The sampling frequency of the input signal.
        graph (bool): If True, plot the resulting Wavelet Transform. Default is False.
        fmax (int): The maximum frequency for the Wavelet Transform. Default is 8192 Hz.

    Returns:
        tfr_morlet (array): The Wavelet Transform of the input signal using the Morlet wavelet.
    """
    
    # Reshape the input signal to have the right dimensions for the time-frequency analysis
    data = data.reshape(1,1,len(data))
    
    # Define the frequency range and the number of cycles per frequency for the Morlet wavelet
    freqs = np.arange(1,fmax,16) #64 niveles de descomposicion
    n_cycles = freqs/8. 
    
    # Compute the Wavelet Transform using the Morlet wavelet
    tF = mne.time_frequency.tfr_array_morlet(epoch_data = data , 
                                             sfreq = sampling_rate , 
                                             freqs = freqs , 
                                             n_cycles = n_cycles,
                                             decim = 128,
                                             verbose = False) 
    # Extract the Wavelet Transform from the output of the mne.time_frequency.tfr_array_morlet function
    tfr_morlet = tF[0,0,:,:]
    
    # Plot the Wavelet Transform if the 'graph' parameter is set to True
    if graph:
        plt.pcolormesh(np.abs(tfr_morlet),cmap='twilight', vmax=abs(tfr_morlet).max(), vmin=-abs(tfr_morlet).max())
        
        plt.title('Wavelet Transform (Morlet) \n freq_lim = ' + str(fmax)
                 )
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        
        plt.show()
    return tfr_morlet

def dwt_pywt(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    """
    Computes the Discrete Wavelet Transform (DWT) of a time series using PyWavelets library.

    Parameters:
        data (ndarray): The time series data.
        sampling_rate (float): The sampling rate of the time series.
        graph (bool): If True, displays the DWT coefficients using a heatmap.
        fmax (int): The maximum frequency to analyze in the DWT.

    Returns:
        The DWT coefficients of the time series.
    """
    freqs = np.arange(1,fmax,16) # Define the frequencies to analyze in the DWT
    
    n_cycles = freqs/8. 
    
    db1 = pywt.Wavelet('db1') # Create the db1 wavelet
    coeff = pywt.wavedec(data, db1) # Compute the DWT coefficients
    
    # Upsample the coefficients to display them using a heatmap
    coeff = np.asarray([np.repeat(c, len(coeff[-1] )/len(c) ) for c in coeff])
    
    # Display the DWT coefficients as a heatmap
    if graph:
        plt.pcolormesh(np.abs(coeff),cmap='twilight', vmax=abs(coeff).max(), vmin=-abs(coeff).max())
        plt.title('dwt_morlet \n freq_lim = ' + str(fmax))
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        
        plt.show()
    return coeff


import librosa
import librosa.display

def mel(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    """
    This function computes the mel spectrogram of an audio signal using librosa.

    Parameters:
        data (ndarray): Audio signal data.
        sampling_rate (int): Sampling rate of the audio signal.
        graph (bool): If True, display the spectrogram plot.
        fmax (int): The maximum frequency (in Hz) of the mel filter bank.

    Returns:
        S (ndarray): The mel spectrogram.
    """
    
    # Define parameters
    n_fft=2024
    hop_length=64
    n_mels=40
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=data, 
                                       sr=sampling_rate, 
                                       n_fft=n_fft, 
                                       hop_length=hop_length,
                                       fmax=fmax,
                                       n_mels=n_mels
                                      )
    # Display the mel spectrogram plot
    if graph:
        
        librosa.display.specshow(S, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='mel',cmap='twilight', vmax=abs(S).max(), vmin=-abs(S).max())
        plt.title('melspectrogram \n freq_lim = ' + str(fmax) 
                 )
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        
        plt.show()
        
    return S


import tftb
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def el_diablo(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    '''
       My PC is struggling to run this, the signal is very large so the function becomes very memory-intensive
    '''
    
    
    #signal = signal[0:15000]
    
    #new_sampling_rate = 1024
    #signal = sig.resample(signal, 2*new_sampling_rate)
    
    T = 2
    # signal duration
    dt = 1/sampling_rate  # sample interval/spacing
    N = T / dt  # number of samples
    ts = np.arange(N) * dt  # times
    
    #  plotting the signal
    #plt.figure()
    #plt.plot(ts, signal)
    #plt.show()
    
    # Doing the WVT
    wvd = tftb.processing.WignerVilleDistribution(data, timestamps=ts)
    tfr_wvd, t_wvd, f_wvd = wvd.run()
    
    # here t_wvd is the same as our ts, and f_wvd are the "normalized frequencies"
    # so we will not use them and construct our own
    #f_wvd = np.fft.fftshift(np.fft.fftfreq(tfr_wvd.shape[0], d=2 * dt))
    #df_wvd = f_wvd[1]-f_wvd[0]
    
    im = plt.imshow(np.fft.fftshift(abs(tfr_wvd), axes=0), vmax=abs(tfr_wvd).max(), vmin=-abs(tfr_wvd).max(),
                 #   extent=(ts[0] - dt/2, 
                 #           ts[-1] + dt/2,
                 #           f_wvd[0]-df_wvd/2, 
                 #           f_wvd[-1]+df_wvd/2),
                    cmap='twilight', aspect='auto', origin='lower')
    plt.ylabel('frequency [Hz]')
    plt.xlabel('time [s]')
    plt.show()
    tfr_wvd.shape


### import numpy as np
import matplotlib.pyplot as plt
import pywt

from scipy import signal
from scipy import optimize


def calcular_cwt(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    """
       This function takes a time series and plots its Continuous Wavelet Transform (CWT) using scipy.
    
       Parameters:
           data (array): the time series.
           sampling_rate (float): the sampling rate of the time series.
           graph (bool): if True, the CWT will be plotted.
           fmax (float): the maximum frequency for the CWT.
    
       Returns:
           The CWT matrix.
    """
    
    # Define the range of scales to compute the CWT
    widths = np.arange(1, 201)
    
    cwtmatr = signal.cwt(data, signal.morlet2, widths)

    # Plot the CWT
    if graph:
        plt.imshow(abs(cwtmatr), cmap = 'twilight', aspect = 'auto', interpolation = 'bilinear' ,
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        
        plt.title('Continuous Wavelet Transform')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        #plt.colorbar()
        plt.show()
    return cwtmatr
    

import ewtpy

def ewt(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    """
    Compute the Empirical Wavelet Transform (EWT) of a time series.

    Parameters:
        data (array): the time series data.
        sampling_rate (float): the sampling rate of the time series.
        graph (bool): whether to plot the EWT.
        fmax (float): the maximum frequency to consider.

    Returns:
        ewt (array): the EWT of the time series.
    """
    # Compute the EWT using the ewtpy package
    ewt, mfb ,boundaries = ewtpy.EWT1D(data, 
                                 N = 5, 
                                 log = 0,
                                 detect = "locmax", 
                                 completion = 0, 
                                 reg = 'average', 
                                 lengthFilter = 10,
                                 sigmaFilter = 5)
    # If graph is True, plot the EWT
    if graph:
        fig, ax = plt.subplots(ewt.shape[1],1)
        fig.tight_layout()
        ax[0].set_title('Empirical Wavelet Transform')
        
        for i in range(ewt.shape[1]):
            ax[i].plot(ewt[:,i])
        
        ax[ewt.shape[1]-1].set_xlabel('Time (s)')
        plt.show()
        
    return ewt 
    

import fcwt
import numpy as np
import matplotlib.pyplot as plt

def calcular_fcwt(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    #Initialize
    fs = sampling_rate
    n = fs*2 #2 seconds
    ts = np.arange(n)
    
    f0 = 1 #lowest frequency
    f1 = fmax #highest frequency
    fn = 200 #number of frequencies
    
    # Calculate the fcwt using the Fast Continuous Wavelet Transform (fcwt) library
    freqs, out = fcwt.cwt(data, fs, f0, f1, fn)
    
    # If graph is True, plot the fcwt
    if graph:
        fcwt.plot(data, fs, f0=f0, f1=f1, fn=fn)
        
    return out

import torch
import numpy as np
from matplotlib import pyplot as plt
from torchHHT import hht, visualization
from scipy.signal import chirp
import IPython

def Hilbert_Huang(data,sampling_rate = 16384 ,graph=False, fmax=8192):
    
    fs = sampling_rate
    duration = 2.0
    t = torch.arange(fs*duration) / fs
 
    x = data
    
    plt.plot(t, x) 
    plt.title("Hilbert_Huang")
    plt.xlabel("time")
    plt.show()
    
    #descomposition
    imfs, imfs_env, imfs_freq = hht.hilbert_huang(x, fs, num_imf=3)
    visualization.plot_IMFs(x, imfs, fs)
    
    #spectrum, t, f = hht.hilbert_spectrum(imfs_env, imfs_freq, fs, freq_lim = (0, fmax), time_scale=1, freq_res = 1)
    #visualization.plot_HilbertSpectrum(spectrum, t, f)
    
    return

from tqdm import tqdm
# Preproceso de datos usando MNE
def prepro_dir(directory, outdir, trns_indx = 0, graph=False, white=False, pad=0, fmax = 8192, verbose = 0):
    """
    Preprocesses a directory of files using one of several possible transformations.

    :param directory: path to directory with input files
    :param outdir: path to directory where output files will be stored
    :param trns_indx: integer representing which transformation to use, as listed in trns_funct and trns_names
    :param graph: unused
    :param white: unused
    :param pad: unused
    :param fmax: frequency threshold in Hz for filtering high frequencies in some transformations
    :param verbose: integer indicating level of detail for printed output
    :return: None
    """
    
    
    # List of possible transformations
    trns_funct = [calcular_stft,wt_morlet,dwt_pywt , calcular_cwt, calcular_fcwt, mel]
    
    # List of names corresponding to each transformation
    trns_names = ['STFT','WT_Morlet','DWT','CWT','fCWT','MELSPECTROGRAM']
    
    # Select the name of the current transformation based on trns_indx
    outdir = outdir + trns_names[trns_indx]
    
    # Print out some details of the current transformation if verbose is True
    if verbose:
        print("\nProcessing " + trns_names[trns_indx])
        print("--------------------------")
        print("\tdir_input: ", directory)
        print("\tdir_output: ", outdir)
        print("\tfmax: ", fmax, " Hz")
        filename = os.listdir(directory)[0]
        f= os.path.join(directory, filename)
        serie_de_tiempo = np.loadtxt(f, delimiter=" ")[:,1]
        dataTransformed = trns_funct[trns_indx](serie_de_tiempo,graph=False , fmax = fmax)
        print("\tshape outdata: ", dataTransformed.shape)
        print("\tdtype outdata: ", dataTransformed.dtype)
        print("--------------------------")
          
    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Iterate through files in the input directory and apply the selected transformation to each one
    for filename in tqdm( os.listdir(directory) ):
        f = os.path.join(directory, filename)
        
        # checking if it is a file
        if os.path.isfile(f):
            
            # Load the data from the file
            data = np.loadtxt(f, delimiter=" ")[:,1]
            
            # Apply the selected transformation to the data
            dataTransformed = trns_funct[trns_indx](data,
                                                    graph=False , 
                                                    fmax = fmax)
            # Absolute value of the transformed data
            dataTransformed = abs(dataTransformed)
            
            # Save the transformed data to a file in the output directory
            outfile =  outdir + '/' + Path(filename).stem + '.txt'
            np.savetxt(outfile, dataTransformed, delimiter=',')
            

def main():
    #here's the use of the function
    # prepro_dir('data/2000 datafull','data/', trns_indx = 0 , fmax = 2048, verbose = 1) #STFT
    # prepro_dir('data/2000 datafull','data/', trns_indx = 1 , fmax = 2048, verbose = 1) #WT Morlet
    # prepro_dir('data/2000 datafull','data/', trns_indx = 2 , fmax = 2048, verbose = 1) #DWT
    # prepro_dir('data/2000 datafull','data/', trns_indx = 3 , fmax = 2048, verbose = 1) #cwt
    # prepro_dir('data/2000 datafull','data/', trns_indx = 4 , fmax = 2048, verbose = 1) #fcwt
    # prepro_dir('data/2000 datafull','data/', trns_indx = 5 , verbose = 1) #mel
   print('Hello World')
if __name__ == '__main__':   
	 main()
     

