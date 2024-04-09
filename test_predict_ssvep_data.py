#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_predict_ssvep_data.py

This script serves as the test script for Project 2: SSVEP. Functions from both import_ssvep_data.py (Lab 3, edited) and predict_ssvep_data.py are called to produce a functional, meaningful script. First, the data are loaded into a dictionary containing features such as EEG voltages and stimulus frequencies. The predictions and figures of merit for a given epoch are then generated initially for one channel and a start/end time pair before being calculated for a variety of possible epochs. Once produced over the different epochs, these data are plotted.

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

@author: Claire Leahy and Lute Lillo
"""

# import packages
import numpy as np
import os
from import_ssvep_data import load_ssvep_data
from predict_ssvep_data import generate_predictions, calculate_figures_of_merit, figures_of_merit_over_epochs, plot_figures_of_merit, plot_predictor_histogram

#%% Load the Data

# Load the SSVEP data for subject 1
data_s1 = load_ssvep_data(subject=1, data_directory='SsvepData/')

# Load the SSVEP data for subject 2
data_s2 = load_ssvep_data(subject=2, data_directory='SsvepData/')

#%% Part A: Generate Predictions

# Generate predictions for subject 1, channel Oz
prediction_quantities_s1, predicted_labels_s1, truth_labels_s1 = generate_predictions(data=data_s1, channel='Oz', epoch_start_time=0, epoch_end_time=20)

# Generate predictions for subject 2, channel Oz
prediction_quantities_s2, predicted_labels_s2, truth_labels_s2 = generate_predictions(data=data_s2, channel='Oz', epoch_start_time=0, epoch_end_time=10)

#%% Part B: Calculate Accuracy and ITR

# Calculate figures of merit for subject 1, channel Oz
accuracy_s1, ITR_time_s1, signal_present_predictors_s1, signal_absent_predictors_s1 = calculate_figures_of_merit(data=data_s1, predicted_labels=predicted_labels_s1, truth_labels=truth_labels_s1, prediction_quantities=prediction_quantities_s1, classes_count=2)

# Calculate figures of merit for subject 2, channel Oz
accuracy_s2, ITR_time_s2, signal_present_predictors_s2, signal_absent_predictors_s2 = calculate_figures_of_merit(data=data_s2, predicted_labels=predicted_labels_s2, truth_labels=truth_labels_s2, prediction_quantities=prediction_quantities_s2, classes_count=2)

#%% Part C: Loop Through Epoch Limits

# Create arrays for start and end times
start_times = np.arange(0,20)
end_times = np.arange(0,20)

# Calculate figures of merit for various epochs for subject 1, channel Oz
figures_of_merit_s1, epoched_predictors_s1 = figures_of_merit_over_epochs(data=data_s1, start_times=start_times, end_times=end_times, channel='Oz')

# Calculate figures of merit for various epochs for subject 2, channel Oz
figures_of_merit_s2, epoched_predictors_s2 = figures_of_merit_over_epochs(data=data_s2, start_times=start_times, end_times=end_times, channel='Oz')

#%% Part D: Plot Results

# Initialize a directory to store plots
if os.path.exists('plots/'): # check if directory exists
    None
else:
    os.mkdir('plots/')

# Plot figures of merit for various epochs for subject 1, channel Oz
plot_figures_of_merit(figures_of_merit_s1, start_times=start_times, end_times=end_times, channel='Oz', subject=1)

# Plot figures of merit for various epochs for subject 2, channel Oz
plot_figures_of_merit(figures_of_merit_s2, start_times=start_times, end_times=end_times, channel='Oz', subject=2)

#%% Part E: Create a Predictor Histogram

# Plot predictor histogram for epoch range for subject 1, channel Oz
plot_predictor_histogram(data_s1, epoch_start_time=5, epoch_end_time=11, channel='Oz', subject=1, threshold=0)

# Plot predictor histogram for epoch range for subject 2, channel Oz
plot_predictor_histogram(data_s2, epoch_start_time=5, epoch_end_time=11, channel='Oz', subject=2, threshold=0)

#%% Part F: Write Up Your Results

# # Create array for subjects
# subjects = [1, 2]

# # Create arrays for start and end times
# start_times = np.arange(0,20)
# end_times = np.arange(0,20)

# # Run for both subjects
# for subject in subjects:
    
#     # Reload the data for present subject (lets cell run independently other than imports)
#     data = load_ssvep_data(subject=subject, data_directory='SsvepData/')
#     channels = data['channels'] # full array of channels for the subject
    
#     # Run for every channel
#     for channel in channels:
        
#         # Figures of merit
#         figures_of_merit, densities = figures_of_merit_over_epochs(data=data, start_times=start_times, end_times=end_times, channel=channel)
        
#         # Plotting figures of merit
#         plot_figures_of_merit(figures_of_merit, start_times=start_times, end_times=end_times, channel=channel, subject=subject)
        
