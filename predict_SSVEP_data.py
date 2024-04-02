#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:35:53 2024

@author: Claire Leahy and Lute Lillo Portero
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum, get_power_spectrum
from filter_ssvep_data import get_envelope, make_bandpass_filter, filter_data

#%% Part A: Generate Predictions

"""

    - For set of epoch FFTs and desired electrode, find elements of FFT representing amplitude at stimuli frequencies
    - Higher amplitude --> Predicted frequency
        - stim1-stim2 as predictor
        - 0 is threshold
    - Generate array of predicted labels, compare to truth labels
    - Flexibiility for electrodes, epoch times, trials, stimulation frequencies
        - Use closest frequencies if none available

"""

def generate_fft_predictions(data, channel_electrode, epoch_start_time=0, epoch_end_time=20):
    
    # extract data
    eeg = data['eeg']*10**6 # convert to µV
    channels = list(data['channels']) # convert to list
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    # Get the stimulus frequencies (sorted low to high)
    stimulus_frequencies = np.unique(event_types) # Shape (2,)
    
    # isolate frequency for high and low
    high_stimulus_frequency = stimulus_frequencies[-1]
    high_stimulus_frequency_int = int(high_stimulus_frequency[:2])
    
    low_stimulus_frequency = stimulus_frequencies[0]
    low_stimulus_frequency_int = int(low_stimulus_frequency[:2])
    
    # Get the filter coeff
    filter_coeff_high = make_bandpass_filter(high_stimulus_frequency_int-1, high_stimulus_frequency_int+1, filter_order=1000, fs=1000)
    filter_coeff_low = make_bandpass_filter(low_stimulus_frequency_int-1, low_stimulus_frequency_int+1, filter_order=1000, fs=1000)
    
    # Get the filtered data for each of the filter coefficients
    filtered_data_high = filter_data(data, filter_coeff_high)
    filtered_data_low = filter_data(data, filter_coeff_low)
    
    # Get predictor based on amplitude of oscillations and epoch the data give the channel of interest, start/end times
    envelope_high = get_envelope(data, filtered_data_high, channel_to_plot=channel_electrode)
    eeg_epochs_high, _, truth_labels = epoch_ssvep_data(data, epoch_start_time, epoch_end_time,
                                                        eeg_data=envelope_high, stimulus_frequency=high_stimulus_frequency)
    eeg_epochs_fft_high, fft_frequencies = get_frequency_spectrum(eeg_epochs_high, fs)  

    envelope_low = get_envelope(data, filtered_data_low, channel_to_plot=channel_electrode)
    eeg_epochs_low, _, truth_labels = epoch_ssvep_data(data, epoch_start_time, epoch_end_time,
                                                       eeg_data=envelope_low, stimulus_frequency=high_stimulus_frequency)

    
    # predict: compare FFT data for the two stimuli
      
    
    predictor = envelope_high - envelope_low
   
    # declare empty array to contain predictions
    predicted_labels = np.zeros(truth_labels.shape)
    
    # Set threshold for comparison
    threshold = 0
    for label_index in range(len(predicted_labels)):
        if predictor > threshold:
            predicted_labels[label_index] = True
        else:
            predicted_labels[label_index] = False
    
    return predicted_labels, truth_labels

#%% Part B: Calculate Accuracy and ITR

"""

    - With true and predicted labels, and epoch timing info, calculate accuracy and ITR (bits/second)

"""

def calculate_figures_of_merit(data, predicted_labels, truth_labels, classes_count=2):
    
    # get timing parameters
    trials_per_second = data['fs'] # sampling frequency
    epoch_count = len(truth_labels) # same as predicted_labels - is this what is meant by "epoch timing info"?
    
    # calculate accuracy
    accuracy = 0
    
    # accuracy = TP + TN/(TP+TN+FP+FN)
    # TP: predicted and truth both True (i.e. 15Hz)
    # TN: predicted and truth both False
    # FP: predicted True, truth False
    # FN: predicted False, truth True
    
    for epoch_index in range(epoch_count):
        
        # true positives
        if predicted_labels[epoch_index] == truth_labels[epoch_index]:
            TP+=1
    
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
