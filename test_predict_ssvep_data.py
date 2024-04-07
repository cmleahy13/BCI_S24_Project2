#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:36:44 2024

@author: Claire Leahy and Lute Lillo
"""

# import packages
from import_ssvep_data import load_ssvep_data
from predict_ssvep_data import generate_predictions, calculate_figures_of_merit, figures_of_merit_over_epochs, plot_figures_of_merit

#%% Load the Data

# Load the SSVEP data for subject 1
data_s1 = load_ssvep_data(subject=1, data_directory='SsvepData/')

# Load the SSVEP data for subject 2
data_s2 = load_ssvep_data(subject=2, data_directory='SsvepData/')

#%% Part A: Generate Predictions

# Generate predictions for subject 1, channel Oz
predicted_labels_s1, truth_labels_s1 = generate_predictions(data=data_s1, channel='Oz', epoch_start_time=0, epoch_end_time=10)

# Generate predictions for subject 2, channel Oz
predicted_labels_s2, truth_labels_s2 = generate_predictions(data=data_s2, channel='Oz', epoch_start_time=0, epoch_end_time=10)

#%% Part B: Calculate Accuracy and ITR

# Calculate figures of merit for subject 1, channel Oz
accuracy_s1, ITR_time_s1 = calculate_figures_of_merit(data_s1, predicted_labels_s1, truth_labels_s1, classes_count=2)

# Calculate figures of merit for subject 2, channel Oz
accuracy_s2, ITR_time_s2 = calculate_figures_of_merit(data_s2, predicted_labels_s2, truth_labels_s2, classes_count=2)

#%% Part C: Loop Through Epoch Limits

# Create arrays for start and end times
start_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
end_times = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Calculate figures of merit for various epochs for subject 1, channel Oz
figures_of_merit_s1 = figures_of_merit_over_epochs(data=data_s1, start_times=start_times, end_times=end_times, channel='Oz')

# Calculate figures of merit for various epochs for subject 2, channel Oz
#figures_of_merit_s2 = figures_of_merit_over_epochs(data=data_s2, start_times=start_times, end_times=end_times, channel='Oz')

#%% Part D: Plot Results

# Plot figures of merit for various epochs for subject 1, channel Oz
plot_figures_of_merit(figures_of_merit_s1, start_times=start_times, end_times=end_times, channel='Oz', subject=1)

# Plot figures of merit for various epochs for subject 2, channel Oz
#plot_figures_of_merit(figures_of_merit_s2, start_times=start_times, end_times=end_times, channel='Oz', subject=2)