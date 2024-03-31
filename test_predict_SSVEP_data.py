#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:36:44 2024

@author: Claire Leahy and Lute Lillo Portero
"""

# import packages
from import_ssvep_data import load_ssvep_data

#%% Load the Data

# load the SSVEP data for subject 1
data_s1 = load_ssvep_data(subject=1, data_directory='SsvepData/')

# load the SSVEP data for subject 2
data_s2 = load_ssvep_data(subject=2, data_directory='SsvepData/')

#%%
