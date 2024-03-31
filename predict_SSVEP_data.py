#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:35:53 2024

@author: Claire Leahy and Lute Lillo Portero
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum

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

def generate_fft_predictions(data, channel, epoch_start_time=0, epoch_end_time=20, stimulus_frequencies=['15hz','12Hz']):
    
    # extract data
    eeg = data['eeg']*10**6 # convert to µV
    channels = list(data['channels']) # convert to list
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    # isolate the stimulus frequency that serves as True
    stimulus_frequency = stimulus_frequencies[0]
    
    # epoch the data give the channel of interest, start/end times
    eeg_epochs, epoch_times, truth_labels = epoch_ssvep_data(data, epoch_start_time, epoch_end_time, stimulus_frequency)
    
    # threshold for amplitude comparisons
    threshold = 0
    
    # predict: compare FFT data for the two stimuli
    eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    
    # compare predicted labels to truth labels
    
    return 

#%% Part B: Calculate Accuracy and ITR

"""

    - With true and predicted labels, and epoch timing info, calculate accuracy and ITR (bits/second)

"""

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
