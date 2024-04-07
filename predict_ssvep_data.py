#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_ssvep_data.py

This script serves as the primary module script for Project 2: SSVEP. In this script, the functions collectively serve to generate predictions for the stimulus frequency given EEG data and subsequently plot these predictions and the associated figures of merit. The generate_predictions() function produces these frequencies given a dictionary of EEG data and associated information about the trials, such as the event stimuli. The produced prediction and truth arrays are evaluated in calculate_figures_of_merit(), where accuracy and the information transfer rate (ITR) in bits per second are calculated. Those same figures of merit are calculated in figures_of_merit_over_epochs(), though this function calculates these values in a loop as multiple epoch lengths (different start and end times) are provided. The figures of merit are then plotted in a pseudocolor plot against the various epoch start and end times. Finally, a predictor histogram is generated using the prediction and truth labels over the different epoch start and end times.

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

@authors: Claire Leahy and Lute Lillo
"""

# import packages
import numpy as np
import warnings
from matplotlib import pyplot as plt
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum

#%% Part A: Generate Predictions

"""

    TODO:
        - frequency dynamic coding
            - find frequencies if not explicitly given (i.e. no event_types) -- this is choosing the closest frequencies if none given
            - assumed that we're taking in a dictionary that contains this field?
            - take two highest-power frequencies (and/or their harmonics with the exception of the powerline artifact)?
        - ignore when an index of power=0
            - gives runtime error but maintains functionality (proceeds to run after error)
            - temporarily using warnings.filterwarnings("ignore", category=RuntimeWarning) to avoid print to console --> likely want to figure out better solution to handle this case
        - Docstrings

"""

def generate_predictions(data, channel='Oz', epoch_start_time=0, epoch_end_time=20):
    """
        Definition:
        ----------
            
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary,
           
            
        Returns:
        ----------
            None
    """
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
    if eeg_epochs is None: # avoid performing FFT on invalid times
        
        # Declare variables for return purposes but give no value
        eeg_epochs_fft = None
        fft_frequencies = None
        predicted_labels = None
    
    else:
        
        # Take FFT of valid epochs
        eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    
        # Find stimuli frequencies indices based on relationship of FFT to time domain
        low_frequency_index = int(((len(fft_frequencies)-1)*2/fs)*low)
        high_frequency_index = int(((len(fft_frequencies)-1)*2/fs)*high)
        
        # Suppress runtime warning that occurs for divide by zero (consequences accounted for elsewhere)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
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
        - need to figure out ITR calculation
            - should the trials_per_second be related to epoch length?
        - Docstrings

"""

def calculate_figures_of_merit(data, predicted_labels, truth_labels, classes_count=2):
    """
        Definition:
        ----------
            
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary,
           
            
        Returns:
        ----------
            None
    """
    
    # Get timing parameters
    trials_per_second = data['fs'] # sampling frequency
    epoch_count = len(truth_labels) # same as predicted_labels
    
    # Assign counters for confusion matrix values
    TP = 0 # true positive initial count
    TN = 0 # true negative initial count
    FP = 0 # false positive initial count
    FN = 0 # false negative initial count
    
    # Carry through errors
    if predicted_labels is None:
        
        # Default accuracy to 0.5, ITR_time to 0.00
        accuracy = 0.5
        ITR_time = 0.00
        
    else:
    
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
        
        # Temporarily rename/replace variables for readability in ITR calculation
        P = accuracy
        N = classes_count
        
        # Calculate ITR
        if P == 1.0:
            ITR_trial = 1.0 # ITR_trial will be 1.0 for an accuracy of 1.0 (cannot substitute P=1.0 into the equation because of mathematical error)
        else:
            ITR_trial = np.log2(N) + P*np.log2(P) + (1-P)*np.log2((1-P)/(N-1)) # bits/epoch
        
        ITR_time = ITR_trial * trials_per_second # bits/second
    
    return accuracy, ITR_time

#%% Part C: Loop Through Epoch Limits

"""

    TODO:
        - get a runtime error related to the call of generate_predictions
            - power conversion to dB is a divide by zero error
            - code continues to run through, likely accounted for elsewhere but not immediately upon occurrence
            - suppressed warning in generate_predictions()
        - at least for channel Oz, there is one epoch (2s, 3s) that is worse that accuracy=0.5 (0.45, worse than guessing)
            - when accuracy is this low, should it be replaced with 0.5 as the minimum placeholder value since this is used for the trials that are not valid?
        - Docstrings

"""

def figures_of_merit_over_epochs(data, start_times, end_times, channel):
    """
        Definition:
        ----------
            
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary,
           
            
        Returns:
        ----------
            None
    """
    
    # Declare list to store label data and figures of merit and labels for each epoch
    figures_of_merit = []
    
    # Perform calculations for each set of valid pairs
    for start in start_times:
        
        for end in end_times:
            
            # Update the list containing the figures of merit
            if end < start: # check to make sure valid start and end time
               
                merit_values = (0.5,0.00) # placeholder value for invalid start/end
                figures_of_merit.append(merit_values) # update list with placeholders
            
            elif ((end - start) > 20) or ((end - start) == 0): # check to make sure times will be within the trial range
                
                merit_values = (0.5,0.00) # placeholder value for invalid start/end
                figures_of_merit.append(merit_values) # update list with placeholders
                
            elif start >= 20: # check that the start time is before end of trial
                merit_values = (0.5,0.00) # placeholder value for invalid start/end
                figures_of_merit.append(merit_values) # update list with placeholders

            else: # times are valid
                
                # Predictions
                predicted_labels, truth_labels = generate_predictions(data, channel, epoch_start_time=start, epoch_end_time=end)
                
                # Accuracy and ITR times
                accuracy, ITR_time = calculate_figures_of_merit(data, predicted_labels, truth_labels)
                merit_values = (accuracy, ITR_time) # tuple containing the accuracy and ITR (bits per second)
                
                # Update list
                figures_of_merit.append(merit_values)
     
    # Convert to an array
    figures_of_merit = np.array(figures_of_merit)
                
    return figures_of_merit

#%% Part D: Plot Results

"""

    TODO:
        - change scale of colorbar
        - Docstrings

"""

def plot_figures_of_merit(figures_of_merit, start_times, end_times, channel, subject):
    """
        Definition:
        ----------
            
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary,
           
            
        Returns:
        ----------
            None
    """
    
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
    figure.colorbar(mappable=None, ax=figure_of_merit_plot[0], label='% Correct')
    figure.colorbar(mappable=None, ax=figure_of_merit_plot[1], label='ITR (bits/sec)')
    # need to add a scale for color bar
        # (50, 100) for accuracy
        # (0, all_ITR_time.max()) for ITR
    figure.tight_layout()
    
    # Save figure
    plt.savefig(f"plots/subject_{subject}_channel_{channel}_figures_of_merit.png")

#%% Part E: Create a Predictor Histogram

"""

    TODO:
        - do we want to plot over a span of epochs?
        - may run into issue with the predicted_labels if the epoch is invalid
            - handled for accuracy and ITR with placeholders --> can we do the same thing here by setting whatever value will be plotted to None or 0 (i.e. have it not contribute to the density)?
        - Docstrings

"""

def plot_predictor_histogram(data, epoch_start_time, epoch_end_time, channel='Oz', subject=1, threshold=0):
    """
        Definition:
        ----------
            
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary,
           
            
        Returns:
        ----------
            None
    """
    
    # Create array of intersection start and end times
    start_times = np.arange(epoch_start_time, epoch_end_time)
    end_times = np.arange(epoch_start_time, epoch_end_time)
    
    # Create empty lists to store data to be plotted
    predictions = []
    truths = []
    
    # Loop through the epochs
    for start in start_times:
        
        for end in end_times:
    
            # Get the predictions for the epoch
            predicted_labels, truth_labels = generate_predictions(data, channel, start, end)
            
            # Update lists
            predictions.append(predicted_labels)
            truths.append(truth_labels)
    
    # Convert lists to arrays
    predictions = np.array(predictions)
    truths = np.array(truths)
    

