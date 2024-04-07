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
from scipy.stats import gaussian_kde
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

"""

def generate_predictions(data, channel='Oz', epoch_start_time=0, epoch_end_time=20):
    """
    Description
    -----------
    Function to generate predictions (quantitatively and qualitatively) about the frequency of the stimulus.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    channel : str, optional
        The electrode for which the labels will be generated. The default is 'Oz'.
    epoch_start_time : int, optional
        The relative time in seconds at which the epoch starts. The default is 0.
    epoch_end_time : int, optional
        The relative time in seconds at which the epoch ends. The default is 20.

    Returns
    -------
    prediction_quantities: array of float, size Ex1 where is the number of epochs
        An array containing the predictor quantities (higher frequency's power minus the lower frequency's power)
    predicted_labels : array of bool, size Ex1 where E is the number of epochs
        An array of boolean type that is True when the magnitude of the power for the higher frequency stimulus is greater than the power of the lower frequency stimulus for the epochs, False if the lower frequency stimulus is greater than the power of the higher frequency stimulus for the epoch.
    truth_labels : array of bool, size Ex1 where E is the number of epochs
        An array containing True if the epoch is actually the higher frequency stimulus, False if the epoch is an event at the lower frequency stimulus. (In other scripts, this variable may be known as is_trial_15Hz.)

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
    
    else: # get data any valid sets
        
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
        
        # Declare empty array to contain prediction data
        prediction_quantities = np.zeros(truth_labels.shape) # numerical values
        predicted_labels = np.empty(truth_labels.shape, dtype=bool) # boolean labels
        
        # Set threshold for comparison
        threshold = 0
        
        # Create and compare predictions
        for epoch_index in range(power_in_dB.shape[0]):
                
            # Generate predictor
            predictor = power_in_dB[epoch_index][channel_index][high_frequency_index] - power_in_dB[epoch_index][channel_index][low_frequency_index]
            
            # Update prediction_quantities with the predictor
            prediction_quantities[epoch_index] = predictor
        
            # Compare predictor to threshold
            if predictor > threshold:
                predicted_labels[epoch_index] = True
            else:
                predicted_labels[epoch_index] = False
    
    return prediction_quantities, predicted_labels, truth_labels

#%% Part B: Calculate Accuracy and ITR

"""

    TODO:
        - need to figure out ITR calculation
            - should the trials_per_second be related to epoch length?
        - Docstrings

"""

def calculate_figures_of_merit(data, predicted_labels, truth_labels, prediction_quantities, classes_count=2):
    
    # Get timing parameters
    trials_per_second = data['fs'] # sampling frequency
    epoch_count = len(truth_labels) # same as predicted_labels
    
    # Assign counters for confusion matrix values
    TP = 0 # true positive initial count
    TN = 0 # true negative initial count
    FP = 0 # false positive initial count
    FN = 0 # false negative initial count
    
    absent = []
    present = []
    
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
                present.append(prediction_quantities[epoch_index])
            elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==False):
                TN += 1 # add to true negative count
                absent.append(prediction_quantities[epoch_index])
            elif (predicted_labels[epoch_index]==True) & (truth_labels[epoch_index]==False):
                FP += 1 # add to false positive count
                absent.append(prediction_quantities[epoch_index])
            elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==True):
                FN += 1 # add to false negative count
                present.append(prediction_quantities[epoch_index])
        
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
    
    return accuracy, ITR_time, present, absent 

#%% Part C: Loop Through Epoch Limits

"""

    TODO:
        - get a runtime error related to the call of generate_predictions
            - power conversion to dB is a divide by zero error
            - code continues to run through, likely accounted for elsewhere but not immediately upon occurrence
            - suppressed warning in generate_predictions()
        - at least for channel Oz, there is one epoch (2s, 3s) that is worse than accuracy=0.5 (0.45, worse than guessing)
            - when accuracy is this low, should it be replaced with 0.5 as the minimum placeholder value since this is used for the trials that are not valid?
        - Docstrings

"""

def figures_of_merit_over_epochs(data, start_times, end_times, channel):
    
    # Declare list to store predictions and figures of merit for each epoch
    figures_of_merit = []

    # Plotting histogram values
    present_den =[]
    absent_den = []
    
    # Perform calculations for each set of valid pairs
    for start in start_times:
        
        for end in end_times:
            
            # Update the list containing the figures of merit
            if end < start: # check to make sure valid start and end time
               
                # Update lists with placeholder values for invalid times
                merit_values = (0.5,0.00)
                figures_of_merit.append(merit_values)
            
            elif ((end - start) > 20) or ((end - start) == 0): # check to make sure times will be within the trial range
                
                # Update lists with placeholder values for invalid times
                
                merit_values = (0.5,0.00)
                figures_of_merit.append(merit_values)
                
            elif start >= 20: # check that the start time is before end of trial
                
                # Update lists with placeholder values for invalid times
                merit_values = (0.5,0.00)
                figures_of_merit.append(merit_values)

            else: # times are valid
                
                # Predictions
                prediction_quantities, predicted_labels, truth_labels = generate_predictions(data, channel, epoch_start_time=start, epoch_end_time=end)
                
                # Figures of merit
                accuracy, ITR_time, present, absent = calculate_figures_of_merit(data, predicted_labels, truth_labels, prediction_quantities)
                merit_values = (accuracy, ITR_time) # tuple containing the accuracy and ITR (bits per second)
                
                # Update lists
                figures_of_merit.append(merit_values)
        
                present_den += present
                absent_den += absent
                
    densities = (present_den, absent_den)
                
    # Convert to arrays
    figures_of_merit = np.array(figures_of_merit)
    densities = (present_den, absent_den)
                
    return figures_of_merit, densities

#%% Part D: Plot Results

"""

    TODO:
        - change scale of colorbar
        - Docstrings

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
    for values in figures_of_merit:
        
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
    plt.close()

#%% Part E: Create a Predictor Histogram

"""

    TODO:
        - handled for accuracy and ITR with placeholders --> can we do the same thing here by setting whatever value will be plotted to None or 0 (i.e. have it not contribute to the density)?
        - Docstrings

"""

def plot_predictor_histogram(densities, channel='Oz', subject=1, threshold=0):
            
    present, absent = densities
 
    """ Plot Present values """
    # Calculate mean and standard deviation
    mean = np.mean(present)
    std_dev = np.std(present)

    # Create bell curve data
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)

    # Calculate kernel density estimate
    kde = gaussian_kde(present)
    density = kde(x)

    # Plot smooth curve and bell curve
    plt.plot(x, density, color='b', label='Smooth curve')  # Plot smooth curve
    plt.fill_between(x, density, color='skyblue', alpha=0.5)

    """ Plot Absent values """
    # Calculate mean and standard deviation
    mean_abs = np.mean(absent)
    std_dev_abs = np.std(absent)

    # Create bell curve data
    x_abs = np.linspace(mean_abs - 3*std_dev_abs, mean_abs + 3*std_dev_abs, 100)

    # Calculate kernel density estimate
    kde = gaussian_kde(absent)
    density_abs = kde(x_abs)

    # Plot smooth curve and bell curve
    plt.plot(x_abs, density_abs, color='r', label='Smooth curve')  # Plot smooth curve
    plt.fill_between(x_abs, density_abs, color='red', alpha=0.5)

    plt.title('Relative Densities of Confusion Matrix Values for Predictors')
    plt.xlabel('Predictors')
    plt.ylabel('Relative Density')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    
    # vertical line at threshold
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
   
    # Save figure
    plt.savefig(f"plots/subject_{subject}_channel_{channel}_prediction_histogram.png")