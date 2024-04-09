#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_ssvep_data.py

This script serves as the primary module script for Project 2: SSVEP. In this script, the functions collectively serve to generate predictions for the stimulus frequency given EEG data and subsequently plot these predictions and the associated figures of merit. The generate_predictions() function produces these frequencies given a dictionary of EEG data and associated information about the trials, such as the event stimuli. The produced prediction and truth arrays are evaluated in calculate_figures_of_merit(), where accuracy and the information transfer rate (ITR) in bits per second are calculated. Those same figures of merit are calculated in figures_of_merit_over_epochs(), though this function calculates these values in a loop as multiple epoch lengths (different start and end times) are provided. The figures of merit are then plotted in a pseudocolor plot against the various epoch start and end times. Finally, a predictor histogram is generated using the prediction and truth labels over the different epoch start and end times. This script relies on modified functions from import_ssvep_data.py (Lab 3).

Useful abbreviations:
    EEG: Electroencephalography
    SSVEP: Steady-state visual evoked potentials
    fs: Sampling frequency
    FFT: Fast Fourier Transform
    FIR: Finite impulse response
    IIR: Infinite impulse response
    TP: True positive - predicted and truth both True (i.e. 15Hz)
    TN: True negative - predicted and truth both False
    FP: False positive - predicted True, truth False
    FN: False negative - predicted False, truth True
    ITR: Information transfer rate
    KDE: Kernel density estimate

@authors: Claire Leahy and Lute Lillo

Soures:
    - ChatGPT to help with handling non-square dimensions for start and end times for plotting
    
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib as mpl
from scipy.stats import gaussian_kde
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum

#%% Part A: Generate Predictions

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

def calculate_figures_of_merit(data, predicted_labels, truth_labels, prediction_quantities, classes_count=2):
    """
    Description
    -----------
    Function that calculates the figures of merit for the data based on the predicted and truth labels.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
     predicted_labels : array of bool, size Ex1 where E is the number of epochs
         An array of boolean type that is True when the magnitude of the power for the higher frequency stimulus is greater than the power of the lower frequency stimulus for the epochs, False if the lower frequency stimulus is greater than the power of the higher frequency stimulus for the epoch.
    truth_labels : array of bool, size Ex1 where E is the number of epochs
        An array containing True if the epoch is actually the higher frequency stimulus, False if the epoch is an event at the lower frequency stimulus. (In other scripts, this variable may be known as is_trial_15Hz.)
    prediction_quantities: array of float, size Ex1 where is the number of epochs
        An array containing the predictor quantities (higher frequency's power minus the lower frequency's power)
    classes_count : int, optional
        The number of different signal types, in this case frequencies, that are being evaluated in the dataset. The default is 2.

    Returns
    -------
    accuracy : float
        The proportion of correct predictions.
    ITR_time : float
        The information transfer rate of the data in bits per second.
    signal_present_predictors : list of float, size E where E is the number of epochs when the signal is present
        A list containing the quantitative predictors (higher frequency power minus lower frequency power) when the signal is present.
    signal_absent_predictors : list of float, size E where E is the number of epochs when the signal is absent
        A list containing the quantitative predictors (higher frequency power minus lower frequency power) when the signal is absent.
    """
    
    
    # Get timing parameters
    trials_per_second = data['fs'] # sampling frequency
    epoch_count = len(truth_labels) # same as predicted_labels
    
    # Assign counters for confusion matrix values
    TP = 0 # true positive initial count
    TN = 0 # true negative initial count
    FP = 0 # false positive initial count
    FN = 0 # false negative initial count
    
    # Declare empty arrays to contain the quantitative predictor based on signal's presence
    signal_absent_predictors = []
    signal_present_predictors = []
    
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
                signal_present_predictors.append(prediction_quantities[epoch_index]) # signal is present
            
            elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==False):
                TN += 1 # add to true negative count
                signal_absent_predictors.append(prediction_quantities[epoch_index]) # signal is absent
            
            elif (predicted_labels[epoch_index]==True) & (truth_labels[epoch_index]==False):
                FP += 1 # add to false positive count
                signal_absent_predictors.append(prediction_quantities[epoch_index]) # signal is absent
            
            elif (predicted_labels[epoch_index]==False) & (truth_labels[epoch_index]==True):
                FN += 1 # add to false negative count
                signal_present_predictors.append(prediction_quantities[epoch_index]) # signal is present
        
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
    
    return accuracy, ITR_time, signal_present_predictors, signal_absent_predictors 

#%% Part C: Loop Through Epoch Limits

def figures_of_merit_over_epochs(data, start_times=np.arange(0,20), end_times=np.arange(0,20), channel='Oz'):
    """
    Description
    -----------
    Function to calculate figures of merit (accuracy, ITR_time), as well as predictors, over a variety of epoch times.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    start_times : array of int, size Tx1 where T is the number of start times to evaluate, optional
        Range of times to investigate as potential epoch start times. The default is np.arange(0,20).
    end_times : array of int, size Tx1 where T is the number of end times to evaluate, optional
        Range of times to investigate as potential epoch end times. The default is np.arange(0,20).
    channel : str, optional
        The electrode for which the epochs will be investigated. The default is 'Oz'

    Returns
    -------
    figures_of_merit : array of float, size TxF where T is the product of the number of start times and number of end times and F is the number of figures of merit being produced (2)
        Array containing the figures of merit, in this case accuracy and ITR (bits/second).
    epoched_predictors : tuple, size 2
        Tuple containing the predictors for each epoch span, containing the predictors (power difference) organized by presence of the signal.

    """
    
    # Declare list to store predictions and figures of merit for each epoch
    figures_of_merit = []

    # Generate empty lists to organize predictors by signal presence
    signal_present =[]
    signal_absent = []
    
    # Perform calculations for each set of valid pairs
    for start in start_times:
        
        for end in end_times:
            
            # Update the list containing the figures of merit
            if (end < start) or ((end - start) > 20) or ((end - start) == 0) or (start >= 20): # check to make sure valid start and end time
               
                # Update lists with placeholder values for invalid times
                merit_values = (0.5,0.00)
                figures_of_merit.append(merit_values)
                
            else: # times are valid
                
                # Predictions
                prediction_quantities, predicted_labels, truth_labels = generate_predictions(data, channel, epoch_start_time=start, epoch_end_time=end)
                
                # Figures of merit
                accuracy, ITR_time, epoched_signal_present_predictors, epoched_signal_absent_predictors = calculate_figures_of_merit(data, predicted_labels, truth_labels, prediction_quantities)
                merit_values = (accuracy, ITR_time) # tuple containing the accuracy and ITR (bits per second)
                
                # Update lists
                figures_of_merit.append(merit_values)
                signal_present += epoched_signal_present_predictors
                signal_absent += epoched_signal_absent_predictors
    
    # Organize predictors by signal presence as tuple
    epoched_predictors = (signal_present, signal_absent)
                
    # Convert figures_of_merit to an array
    figures_of_merit = np.array(figures_of_merit)
                
    return figures_of_merit, epoched_predictors

#%% Part D: Plot Results

def plot_figures_of_merit(figures_of_merit, start_times, end_times, channel='Oz', subject=1):
    """
    Description
    -----------
    Function to generate pseudocolor plots (confusion matrix) of accuracy and ITR_time against different start and end epoch times.

    Parameters
    ----------
    figures_of_merit : array of float, size TxF where T is the product of the number of start times and number of end times and F is the number of figures of merit (2)
        Array containing the figures of merit, in this case accuracy and ITR (bits/second).
    start_times : array of int, size Tx1 where T is the number of start times to evaluate
        Range of times to investigate as potential epoch start times.
    end_times : array of int, size Tx1 where T is the number of end times to evaluate
        Range of times to investigate as potential epoch end times.
    channel : str, optional
        The electrode for which the figures of merit will be plotted. The default is 'Oz'.
    subject : int, optional
        The subject for which the figures of merit will be plotted. The default is 1.

    Returns
    -------
    None.

    """

    # Convert start and end times lists to arrays for plotting
    start_times = np.array(start_times)
    end_times = np.array(end_times)
    
    # Counts of start/end times to reshape grids
    # start_times, end_times = valid_range_times(start_times, end_times)
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
    if start_times_count != end_times_count:
         start_times_grid, end_times_grid, = np.meshgrid(start_times, end_times) # used ChatGPT to help troubleshoot uneven start and end times dimensions mismatch
    else:
        end_times_grid, start_times_grid = np.meshgrid(end_times, start_times)
    #end_times_grid, start_times_grid = np.meshgrid(end_times, start_times)
    
    # Convert lists to arrays
    all_accuracies = np.array(all_accuracies)
    all_ITR_time = np.array(all_ITR_time)
    
    # Reshape arrays to match grid shape
    all_accuracies = all_accuracies.reshape(end_times_count, start_times_count)
    all_ITR_time = all_ITR_time.reshape(end_times_count, start_times_count)
    
    
    # Plot the figures of merit over epoch lengths
    figure_of_merit_plot[0].pcolor(end_times_grid, start_times_grid, all_accuracies, cmap='viridis')
    figure_of_merit_plot[1].pcolor(end_times_grid, start_times_grid, all_ITR_time, cmap='viridis')
    
    # Format and stylize figure
    # Subplot labels
    figure_of_merit_plot[0].set_title('Accuracy')
    figure_of_merit_plot[1].set_title('Information Transfer Rate')
    
    # Format both plots
    for plot in figure_of_merit_plot:
        
        # Grid
        plot.grid()
        
        # Limits
        plot.set_xlim(end_times.min(), end_times.max())
        plot.set_ylim(start_times.min(), start_times.max()) 
        
        # Labels
        plot.set_xlabel('Epoch End Time (s)')
        plot.set_ylabel('Epoch Start Time (s)')

    
    # Color bars
    norm_accuracy = mpl.colors.Normalize(vmin=all_accuracies.min()*100, vmax=all_accuracies.max()*100)
    figure.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm_accuracy), ax=figure_of_merit_plot[0], label='% Correct')
    
    norm_ITR = mpl.colors.Normalize(vmin=all_ITR_time.min(), vmax=all_ITR_time.max())
    figure.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm_ITR), ax=figure_of_merit_plot[1], label='ITR (bits/sec)')
        
    # Format whole figure
    figure.suptitle(f'SSVEP Subject {subject}, Channel {channel}')
    figure.tight_layout()
    
    # Save figure
    plt.savefig(f"plots/subject_{subject}_channel_{channel}_figures_of_merit.png")

#%% Part E: Create a Predictor Histogram

def plot_predictor_histogram(data, epoch_start_time, epoch_end_time, channel='Oz', subject=1, threshold=0):
    """
    Description
    -----------
    Function to plot a prediction histogram depicting the relative density of the true signal presence or absence compared to the power predictor.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    epoch_start_time : int, optional
        The relative time in seconds at which the epoch starts. The default is 0.
    epoch_end_time : int, optional
        The relative time in seconds at which the epoch ends. The default is 20.
    channel : str, optional
        The electrode for which the densities will be plotted. The default is 'Oz'.
    subject : int, optional
        The relative time in seconds at which the epoch ends. The default is 1.
    threshold : int, optional
        The quantitative difference between the high and low frequency stimuli's powers that defines the predicted labels. The default is 0.

    Returns
    -------
    None.

    """
        
    # Create arrays to get the start through end range
    start_times = np.arange(epoch_start_time, epoch_end_time)
    end_times = np.arange(epoch_start_time, epoch_end_time)
    
    # Get the predictors for the range
    _, predictors_by_presence = figures_of_merit_over_epochs(data, start_times=start_times, end_times=end_times, channel='Oz')
    signal_present, signal_absent = predictors_by_presence # unpack tuple
 
    # Calculate mean and standard deviation of predictor when signal present
    mean_present = np.mean(signal_present)
    standard_deviation_present = np.std(signal_present)
    
    # Calculate mean and standard deviation of predictor when signal absent
    mean_absent = np.mean(signal_absent)
    standard_deviation_absent = np.std(signal_absent)

    # Create bell curve data
    x_present = np.linspace(mean_present - 3*standard_deviation_present, mean_present + 3*standard_deviation_present, 100)
    
    # Create bell curve data
    x_absent = np.linspace(mean_absent - 3*standard_deviation_absent, mean_absent + 3*standard_deviation_absent, 100)

    # Calculate kernel density estimate when signal is present
    kde_present = gaussian_kde(signal_present)
    density_present = kde_present(x_present)

    # Calculate kernel density estimate when signal is absent
    kde_absent = gaussian_kde(signal_absent)
    density_absent = kde_absent(x_absent)

    # Plotting
    # Declare a new figure
    plt.figure()
    
    # Plot densities
    plt.fill_between(x_present, density_present, color='skyblue', alpha=0.5, label='present')
    plt.fill_between(x_absent, density_absent, color='red', alpha=0.5, label='absent')
    
    # Vertical line at threshold
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label='Threshold')

    # Format figure
    plt.title(f'Prediction Histogram for Subject {subject}, Channel {channel}')
    plt.xlabel('Predictors')
    plt.ylabel('Relative Density')
    plt.xticks(rotation=45) # rotate x-axis labels for better readability
    plt.legend()
    plt.grid()
    plt.tight_layout()
   
    # Save figure
    plt.savefig(f"plots/subject_{subject}_channel_{channel}_prediction_histogram.png")