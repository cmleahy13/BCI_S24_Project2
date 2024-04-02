#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:35:53 2024

Useful abbreviations:
    EEG: electroencephalography
    SSVEP: steady-state visual evoked potentials
    fs: sampling frequency
    FFT: Fast Fourier Transform
    FIR: Finite impulse response
    IIR: Infinite impulse response
    TP: True positive - predicted and truth both True (i.e. 15Hz)
    TN: True negative - predicted and truth both False
    FP: False positive - predicted True, truth False
    FN: False negative - predicted False, truth True
    ITR: Information transfer rate

@author: Claire Leahy and Lute Lillo
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum, plot_power_spectrum
from filter_ssvep_data import make_bandpass_filter, filter_data, get_envelope

#%% Part A: Generate Predictions

"""

    TODO:
        - frequency dynamic coding
            - find frequencies if not explicitly given (i.e. no event_types) -- this is choosing the closest frequencies if none given
                - basically how we'd go about finding the stimuli if only EEG data were given
            - possibly look at an occipital channel power spectrum, take two highest-power frequencies (and/or their harmonics with the exception of the powerline artifact) and use those?
        - make channel inputs more dynamic?
            - should user have option to evaluate multiple channels at one time? lab suggests otherwise
        - Docstrings

"""

def generate_predictions(data, channel='Oz', epoch_start_time=0, epoch_end_time=20):
    
    # extract necessary data
    channels = list(data['channels']) # convert to list
    fs = data['fs']
    event_types = data['event_types']
    
    # find the frequencies of interest
    frequencies = np.unique(event_types) # gets the different stimulus frequencies
    low_frequency, high_frequency = frequencies # assuming 2 stimuli
    low = int(low_frequency[:2]) # gets the integer value of the frequency
    high = int(high_frequency[:2]) # gets the integer value of the frequency
    
    # epoch the data
    eeg_epochs, epoch_times, truth_labels = epoch_ssvep_data(data, epoch_start_time, epoch_end_time, eeg_data=None, stimulus_frequency=high_frequency) # truth_labels will contain True if epoch is the higher stimulus frequency
    
    # take the FFT data of the epochs
    eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    
    # find the indices of the frequencies of the stimuli
    # equation derived from the relationship of FFT to time domain
    low_frequency_index = int(((len(fft_frequencies)-1)*2/fs)*low)
    high_frequency_index = int(((len(fft_frequencies)-1)*2/fs)*high)
    
    # calculate the power
    power = (np.abs(eeg_epochs_fft[:,:,:]))**2 # calculate power for each frequency at each electrode
    
    # get channel index for electrode of interest
    channel_index = channels.index(channel)
    
    # declare empty arrays to contain prediction data
    predictor_array = np.zeros(power.shape[0])
    predicted_labels = np.empty(truth_labels.shape, dtype=bool)
    
    # set threshold for comparison
    threshold = 0
    
    for epoch_index in range(power.shape[0]):
            
        # generate predictor
        predictor = power[epoch_index][channel_index][high_frequency_index] - power[epoch_index][channel_index][low_frequency_index]
        
        # fill in predictor array
        predictor_array[epoch_index] = predictor
    
        # compare predictor to threshold
        if predictor > threshold:
            predicted_labels[epoch_index] = True
        else:
            predicted_labels[epoch_index] = False
    
    return predicted_labels, truth_labels

#%% Part B: Calculate Accuracy and ITR

"""

    TODO:
        - is epoch timing info really only the epoch count? 
            - epoch start and end times affect this variable
            - ITR_trial does look like it matches up with graphic in class for 2 classes
        - double check that trials_per_second should be fs
        - Docstrings

"""

def calculate_figures_of_merit(data, predicted_labels, truth_labels, classes_count=2):
    
    # get timing parameters
    trials_per_second = data['fs'] # sampling frequency
    epoch_count = len(truth_labels) # same as predicted_labels - is this what is meant by "epoch timing info"?
    
    # assign counters for confusion matrix values
    TP = 0 # true positive initial count
    TN = 0 # true negative initial count
    FP = 0 # false positive initial count
    FN = 0 # false negative initial count
    
    # compare the truth label to the predicted label for each epoch
    for epoch_index in range(epoch_count):
        
        if (predicted_labels[epoch_index]==True) & (truth_labels[epoch_index]==True):
            TP+=1 # add to true positive count
        elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==False):
            TN+=1 # add to true negative count
        elif (predicted_labels[epoch_index]==True) & (truth_labels[epoch_index]==False):
            FP+=1 # add to false positive count
        elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==True):
            FN+=1 # add to false negative count
        
    # calculate accuracy
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    # calculate ITR
    ITR_trial = np.log2(classes_count) + accuracy*np.log2(accuracy) + (1-accuracy)*np.log2((1-accuracy)/(classes_count-1)) # bits/epoch
    
    ITR_time = ITR_trial * trials_per_second # bits/second
    
    return accuracy, ITR_time

#%% Part C: Loop Through Epoch Limits

"""

    - Loop to test set of epoch start/end times
    - Epoch data, calculate FFT, predict, calculate figures of merit
    - Allow any possible start/end time

"""

#%% Part D: Plot Results

"""

    - Generate pseudocolor plots to evalute accuracy, ITR for epochs
    - Allow any possible start/end time
    - Run code for both subjects

"""

#%% Part E: Create a Predictor Histogram

"""

    - Select epoch start/end time that produces high ITR but doesn't have perfect accuracy
    - Calculate predictor (amp15-amp12) variable from given times for each epoch, plot as predictor histogram
    - Use to place threshold

"""
