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
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum

#%% Part A: Generate Predictions

"""

    TODO:
        - frequency dynamic coding
            - find frequencies if not explicitly given (i.e. no event_types) -- this is choosing the closest frequencies if none given
            - take two highest-power frequencies (and/or their harmonics with the exception of the powerline artifact)?
        - make sure epoch start and end times are valid
            - from epoch_ssvep_data() in import_ssvep_data.py
            - works with separate times, but want to make sure the epochs aren't strictly limited to the events
                - want to check ability to predict accurately with shorter epochs
        - Docstrings

"""

def generate_predictions(data, channel='Oz', epoch_start_time=0, epoch_end_time=20):
    
    # Extract necessary data
    channels = list(data['channels']) # convert to list
    fs = data['fs']
    event_types = data['event_types']
    
    # Find the frequencies of interest
    frequencies = np.unique(event_types) # gets the different stimulus frequencies
    low_frequency, high_frequency = frequencies # assuming 2 stimuli
    low = int(low_frequency[:2]) # gets the integer value of the frequency
    high = int(high_frequency[:2]) # gets the integer value of the frequency
    
    # Epoch the data
    eeg_epochs, epoch_times, truth_labels = epoch_ssvep_data(data, epoch_start_time, epoch_end_time, eeg_data=None, stimulus_frequency=high_frequency) # truth_labels contains True if epoch is the higher stimulus frequency
    
    # Take the FFT data of the epochs
    eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    
    # Find stimuli frequencies indices based on relationship of FFT to time domain
    low_frequency_index = int(((len(fft_frequencies)-1)*2/fs)*low)
    high_frequency_index = int(((len(fft_frequencies)-1)*2/fs)*high)
    
    # Calculate the power (dB)
    power = (np.abs(eeg_epochs_fft[:,:,:]))**2 # calculate power for each frequency at each electrode
    power_in_dB = 10*np.log10(power) # convert to dB
    
    # Get channel index for electrode of interest
    channel_index = channels.index(channel)
    
    # Declare empty array to contain prediction labels
    predicted_labels = np.empty(truth_labels.shape, dtype=bool)
    
    # Set threshold for comparison
    threshold = 0
    
    # Create and compare predictions
    for epoch_index in range(power_in_dB.shape[0]):
            
        # Generate predictor
        predictor = power_in_dB[epoch_index][channel_index][high_frequency_index] - power_in_dB[epoch_index][channel_index][low_frequency_index]
    
        # Compare predictor to threshold
        if predictor > threshold:
            predicted_labels[epoch_index] = True
        else:
            predicted_labels[epoch_index] = False
    
    return predicted_labels, truth_labels

#%% Part B: Calculate Accuracy and ITR

"""

    TODO:
        - double check that trials_per_second should be fs
        - Docstrings

"""

def calculate_figures_of_merit(data, predicted_labels, truth_labels, classes_count=2):
    
    # Get timing parameters
    trials_per_second = data['fs'] # sampling frequency
    epoch_count = len(truth_labels) # same as predicted_labels
    
    # Assign counters for confusion matrix values
    TP = 0 # true positive initial count
    TN = 0 # true negative initial count
    FP = 0 # false positive initial count
    FN = 0 # false negative initial count
    
    # Compare the truth label to the predicted label for each epoch
    for epoch_index in range(epoch_count):
        
        if (predicted_labels[epoch_index]==True) & (truth_labels[epoch_index]==True):
            TP += 1 # add to true positive count
        elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==False):
            TN += 1 # add to true negative count
        elif (predicted_labels[epoch_index]==True) & (truth_labels[epoch_index]==False):
            FP += 1 # add to false positive count
        elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==True):
            FN += 1 # add to false negative count
        
    # Calculate accuracy
    accuracy = (TP+TN)/epoch_count
    
    # Temporarily rename variables for readability in ITR calculation
    P = 0.99999 if accuracy == 1.0 else accuracy # handles mathematical error in ITR calculation
    N = classes_count
    
    # Calculate ITR
    ITR_trial = np.log2(N) + P*np.log2(P) + (1-P)*np.log2((1-P)/(N-1)) # bits/epoch
    ITR_time = ITR_trial * trials_per_second # bits/second
    
    return accuracy, ITR_time

#%% Part C: Loop Through Epoch Limits

"""

    TODO:
        - EVERY EPOCH IS RETURNING THE SAME VALUE FOR A GIVEN CHANNEL!!!

"""

def figures_of_merit_over_epochs(data, start_times, end_times, channel):
    
    # Declare list to store label data and figures of merit for each epoch
    figures_of_merit = []
    
    # Perform calculations for each set of valid pairs
    for start in start_times:
        
        for end in end_times:
            
            if start < end:  # check for validity of pair
                
                # Predictions
                predicted_labels, truth_labels = generate_predictions(data, channel, epoch_start_time=start, epoch_end_time=end)
                
                # Accuracy and ITR times
                accuracy, ITR_time = calculate_figures_of_merit(data, predicted_labels, truth_labels)
                merit_values = (accuracy, ITR_time) # tuple containing the accuracy and ITR (bits per second) for the labels
                
                # Update list with the merit values for the epoch
                figures_of_merit.append(merit_values)
            
            else:
                print(f"Start time {start}s and end time {end}s are not a possible combination")
                figures_of_merit.append([0.5,0]) # placeholder value for invalid start-end combinations should be 50% accuracy (guessing), 0 ITR (no information transferred) 
     
    # Convert to an array
    figures_of_merit = np.array(figures_of_merit)
                
    return figures_of_merit

#%% Part D: Plot Results

"""

    TODO:
        - add colorbar (https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot)
        - yticks not showing for ITR subplot

"""

def plot_figures_of_merit(figures_of_merit, start_times, end_times, channel, subject):

    # Convert start and end times lists to arrays for plotting
    start_times = np.array(start_times)
    end_times = np.array(end_times) 
    
    start_times_count = len(start_times)
    end_times_count = len(end_times)

    # Declare lists to contain figures of merit for use in plotting
    all_accuracies = []
    all_ITR_time = []
    
    # Unload data from figures of merit to plot
    for values in  figures_of_merit:
        
        # Get figures of merit from the array
        accuracy = values[0] # first value in tuple is accuracy
        ITR_time = values[1] # second value in tuple is ITR
        
        # Update lists
        all_accuracies.append(accuracy)
        all_ITR_time.append(ITR_time)
    
    # Initialize figure
    figure, figure_of_merit_plot = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
    
    # Update start and end times as grid
    end_times_grid, start_times_grid = np.meshgrid(end_times, start_times)
    
    # Convert lists to arrays
    all_accuracies = np.array(all_accuracies)
    all_ITR_time = np.array(all_ITR_time)
    
    # Reshape arrays to match grid shape
    all_accuracies = all_accuracies.reshape(end_times_count, start_times_count)
    all_ITR_time =all_ITR_time.reshape(end_times_count, start_times_count)
    
    # Plot the figures of merit over epoch lengths
    figure_of_merit_plot[0].pcolor(end_times_grid, start_times_grid, all_accuracies, cmap='viridis')
    figure_of_merit_plot[1].pcolor(end_times_grid, start_times_grid, all_ITR_time, cmap='viridis')
    
    # Format and stylize figure
    # Format accuracy subplot
    figure_of_merit_plot[0].grid()
    figure_of_merit_plot[0].set_title('Accuracy')
    figure_of_merit_plot[0].set_xlabel('Epoch End Time (s)')
    figure_of_merit_plot[0].set_ylabel('Epoch Start Time (s)')
    figure_of_merit_plot[0].set_ylim(0, start_times.max())
    figure_of_merit_plot[0].set_yticks(np.arange(start_times.min(), start_times.max(), 2.5))
    
    # Format ITR subplot
    figure_of_merit_plot[1].grid()
    figure_of_merit_plot[1].set_title('Information Transfer Rate')
    figure_of_merit_plot[1].set_xlabel('Epoch End Time (s)')
    figure_of_merit_plot[1].set_ylabel('Epoch Start Time (s)')
    figure_of_merit_plot[1].set_ylim(0, start_times.max())
    figure_of_merit_plot[1].set_yticks(np.arange(start_times.min(), start_times.max(), 2.5))
    
    # Format whole figure
    figure.suptitle(f'SSVEP Subject {subject}, Channel {channel}')
    figure.tight_layout()
    
    plt.savefig(f"subject_{subject}_channel_{channel}_figures_of_merit.png")

#%% Part E: Create a Predictor Histogram

"""

    - Select epoch start/end time that produces high ITR but doesn't have perfect accuracy
    - Calculate predictor (amp15-amp12) variable from given times for each epoch, plot as predictor histogram
    - Use to place threshold

"""
