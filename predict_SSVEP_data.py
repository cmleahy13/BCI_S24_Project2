#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:35:53 2024

@author: Claire Leahy and Lute Lillo Portero
"""

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
