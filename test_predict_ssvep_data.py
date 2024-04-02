#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:36:44 2024

@author: Claire Leahy and Lute Lillo
"""

# import packages
from import_ssvep_data import load_ssvep_data
from predict_ssvep_data import generate_predictions

#%% Load the Data

# load the SSVEP data for subject 1
data_s1 = load_ssvep_data(subject=1, data_directory='SsvepData/')

# load the SSVEP data for subject 2
data_s2 = load_ssvep_data(subject=2, data_directory='SsvepData/')

#%% Part A: Generate Predictions

# generate predictions for subject 1
predicted_labels_s1, truth_labels_s1 = generate_predictions(data=data_s1, channel='Oz', epoch_start_time=0, epoch_end_time=20)

# generate predictions for subject 2
predicted_labels_s2, truth_labels_s2 = generate_predictions(data=data_s2, channel='Oz', epoch_start_time=0, epoch_end_time=20)
