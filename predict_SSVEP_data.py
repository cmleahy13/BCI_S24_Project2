#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:35:53 2024

@author: Claire Leahy and Lute Lillo Portero
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum, plot_power_spectrum, get_power_spectrum

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
    
    # get the stimulus frequencies (sorted low to high)
    stimulus_frequencies = np.unique(event_types)
    
    # isolate frequency that serves as True (index 1 is the higher frequency)
    stimulus_frequency = stimulus_frequencies[1]
    
    # epoch the data give the channel of interest, start/end times
    eeg_epochs, epoch_times, truth_labels = epoch_ssvep_data(data, epoch_start_time, epoch_end_time, stimulus_frequency)
    
    # threshold for amplitude comparisons
    threshold = 0
    
    # predict: compare FFT data for the two stimuli
    eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    # calculate power spectrum - alter function in import_ssvep_data
    # do we go as far as comparing the envelopes?
    
    # TODO: Try - Catch error for channels to be of type = List()
    spectrum_db_12Hz, spectrum_db_15Hz = get_power_spectrum(eeg_epochs_fft, is_trial_15Hz=truth_labels, channels=channel_electrode)
    
    # compare predicted labels to truth labels for each epoch
    predicted_labels = np.zeros(truth_labels.shape) # declare empty array to contain predictions
    #for label_index in range(len(predicted_labels)):
        # if stim15-stim12 > 0:
            # predicted_labels[label_index] = True
        # else:
            # predicted_labels[label_index] = False
    
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
