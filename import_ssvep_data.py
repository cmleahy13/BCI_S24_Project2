#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:00:05 2024

import_ssvep_data.py

This file serves as the module script for BCI Spring 2024 Lab 03. Below, several functions are defined with the ultimate goals of loading and plotting SSVEP data in both the time and frequency domains. load_ssvep_data() allows the user to load in all data relevant to this lab, the return of which is utilized across multiple of the remaining functions. The plot_raw_data() function plots aspects of the data, such as EEG, in the time domain, as well as identifies the frequency of the sample. epoch_ssvep_data() isolates fragments of the data based on periods of time in which the sample is experiencing a 12Hz or 15Hz trial. Finally, get_frequency_spectrum() and plot_power_spectrum() are two functions that are used to convert the aforementioned data into the frequency domain and plot it.

Useful abbreviations:
    EEG: electroencephalography
    SSVEP: steady-state visual evoked potentials
    fs: sampling frequency
    FFT: Fast Fourier Transform

@authors: Peijin Chen and Claire Leahy ***UPDATED BY RON BRYANT AND CLAIRE LEAHY***

Sources:

    - Improving efficiency of updating an array: ChatGPT
    - Understanding vectorization to improve efficiency: https://moez-62905.medium.com/mastering-numpy-tips-and-tricks-for-efficient-numerical-computing-624d44b4bebd
    - Understanding enumerate to improve efficiency: https://blog.hubspot.com/website/python-enumerate
    - Avoiding printing "dict_keys" data type: https://blog.finxter.com/python-print-dictionary-keys-without-dict_keys/
    - Setting axis tickmarks: ChatGPT
    - Adding tickmarks when sharex=True: https://stackoverflow.com/questions/4209467/matplotlib-share-x-axis-but-dont-show-x-axis-tick-labels-for-both-just-one
    
"""

# import packages
import numpy as np
from matplotlib import pyplot as plt

#%% Part 1: Load the Data
def load_ssvep_data(subject, data_directory):
    """
    Description
    -----------
    Function to load in the SSVEP data from Python's MNE dataset as a dictionary.

    Parameters
    ----------
    subject : int
        The subject for which the data will be loaded.
    data_directory : string,
        The local directory in which the SSVEP data is contained.

    Returns
    -------
    data_dict : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.

    """
    
    # load data
    data = np.load(f'{data_directory}SSVEP_S{subject}.npz', allow_pickle=True) # lib.npyio.NpzFile data type
    
    # explicitly convert from existing data type to dict to avoid potential complications
    data_dict = {'eeg': data['eeg'], 'channels': data['channels'], 'fs': data['fs'], 'event_samples': data['event_samples'], 'event_durations': data['event_durations'], 'event_types': data['event_types']}
    
    # printing to inform user of some data features
    print(f'Data keys: {list(data_dict.keys())}') # list conversion prevents printing data type, adapted from blog.finxter.com (list cast)
    print('\nChannels: ', data['channels'])
    print('\nSampling frequency (Hz):', data['fs'], '\n')
    
    # return data dictionary
    return data_dict

#%% Part 2: Plot the Data

def plot_raw_data(data, subject, channels_to_plot):
    """
    Description
    -----------
    Function that plots the EEG data and the event type (12Hz or 15Hz) in the time domain.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    subject : int
        The subject for which the data will be plotted.
    channels_to_plot : list, size Cx1, where C is the number of channels to be plotted
        Input containing which channels will be plotted.

    Returns
    -------
    None.

    """
    
    # extract data
    eeg = data['eeg']
    channels = list(data['channels'])
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    # find all sample times
    # code adapted from ChatGPT based on our original code (efficiency)
    t = np.arange(len(eeg.T))*(1/fs) # number of samples spaced by 1, subsequently converted to time
    
    # find event samples and times
    # code adapted from ChatGPT based on our original code (efficiency)
    event_intervals = np.zeros([len(event_samples),2]) # array to contain interval times
    # find the end sample indices of the events
    event_ends = event_samples + event_durations # event_samples contains the start samples
    # for each event (:), 0 is the interval start time, and 1 is the interval end time
    event_intervals[:,0] = event_samples/fs # convert start samples to times
    event_intervals[:,1] = event_ends/fs # convert end samples to times
    
    # inform user of plotting
    print('\nPlotting raw data...')
    
    # initialize figure
    figure, sub_figure = plt.subplots(2, sharex=True)
    
    # top subplot containing flash frequency over span of event
    for event_number, interval in enumerate(event_intervals):
    
        # determine the event frequency to plot
        if event_types[event_number] == "12hz":
            event_frequency = 12
    
        else: 
            event_frequency = 15
        
        # plottting the event frequency
        sub_figure[0].hlines(xmin=interval[0], xmax=interval[1], y=event_frequency, label='o') # line
        sub_figure[0].plot([interval[0], interval[1]], [event_frequency,event_frequency], 'bo') # start and end markers
    
    # bottom subplot contain raw data from specified electrodes
    for channel_number, channel_name in enumerate(channels_to_plot):
      
        # identify EEG data to plot
        channel_index = channels.index(channel_name) # index of channel of interest considering all channels      
        eeg_data = data['eeg'][channel_index] # EEG data for the channel
    
        # plot EEG data in µV from channel
        sub_figure[1].plot(t, eeg_data*(10**6), label=channel_name) 
    
    # format figure
    # subplot 1: sample frequency at given time
    sub_figure[0].set_xlabel('time(s)')
    sub_figure[0].set_ylabel('Flash frequency')
    sub_figure[0].set_yticks([12,15])
    sub_figure[0].set_yticklabels(['12Hz','15Hz']) # made aware of set_yticklabels() existence by ChatGPT
    sub_figure[0].grid()
    
    # subplot 2: EEG data over time
    sub_figure[1].set_xlabel('time (s)')
    sub_figure[1].set_ylabel('Voltage (µV)')
    sub_figure[1].legend(loc='best') # place legend in best location given data
    sub_figure[1].grid()
    
    # set plot title
    figure.suptitle(f'SSVEP Subject {subject} Raw Data')
    
    # general appearance
    figure.tight_layout()
    
    # save image
    plt.savefig(f'SSVEP_S{subject}_rawdata.png')
    
    # inform user of plotting completion
    print('Plotting raw data complete.')

#%% Part 3: Extract the Epochs

def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20, eeg_data=None):
    """
    Description
    -----------
    Function that takes in the data dictionary as well as relative start and end times for a given epoch to organize the EEG data by channel and event type.
    
    ***UPDATES***
    Optional input of eeg_data added (default is None). EEG data has been converted to microvolts where applicable. epoch_times has been corrected (no longer uses linspace).

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    eeg_data : array of floats, size CxS, where C is the number of channels and S is the number of samples
        Array containing a form of EEG data in volts.
    epoch_start_time : int, optional
        The relative time in seconds at which the epoch starts. The default is 0.
    epoch_end_time : int, optional
        The relative time in seconds at which the epoch ends. The default is 20.
    eeg_data : array of floats, size CxS, where C is the number of channels and S is the number of samples, optional
        An explicit entry of the EEG data that will be epoched. The default is None.

    Returns
    -------
    eeg_epochs : array of floats, size ExCxS, where E is the number of epochs, C is the number of channels, and S is the number of samples within the epoch
        Array containing the EEG data in volts from each of the electrode channels organized by periods of time in which an event (12Hz or 15Hz flashes) occurs.
    epoch_times : array of floats, size Sx1, where S is the number of samples within each epoch
        Array containing the relative times in seconds of each sample within an epoch.
    is_trial_15Hz : array of boolean, size Ex1, where E is the number of epochs (or events)
        Array containing True if the epoch is an event at 15Hz, False if the epoch is an event at 12Hz.

    """
  
    # extract data
    if eeg_data is None:
        eeg_data = (data_dict['eeg'])*(10**6) # defaulting with dictionary extraction, converted to µV    
    channels = list(data_dict['channels']) # convert to list
    fs = data_dict['fs']
    event_durations = data_dict['event_durations'].astype(int) # cast to contain int instead of float
    event_samples = data_dict['event_samples']
    event_types = data_dict['event_types']
    samples_per_epoch = int(fs*(epoch_end_time-epoch_start_time)) # convert to int
    
    # preallocate array to contain epochs
    eeg_epochs = np.zeros([len(event_samples),len(channels), samples_per_epoch])
    
    # load data into array by epoch
    # code adapted from ChatGPT based on our original code (efficiency)
    for epoch, start_index in enumerate(event_samples): # epoch number is 1st value, content of event_samples at epoch number (i.e. the starting index) is 2nd
    
        end_index = start_index + event_durations[epoch] # find the final sample index for an epoch
        
        eeg_epochs[epoch] = eeg_data[:,start_index:end_index] # for the epoch, add EEG data from all channels (:) for every sample between the start and end indices (start_index:end_index)
            
    # create array containing the times for each sample in the epoch
    epoch_times = np.arange(epoch_start_time, epoch_end_time, 1/fs)
    
    # create boolean array containing True if the event is a 15Hz sample, False if 12Hz
    is_trial_15Hz = np.array([True if event == '15hz' else False for event in event_types])
    
    return eeg_epochs, epoch_times, is_trial_15Hz

#%% Part 4: Take the Fourier Transform

def get_frequency_spectrum(eeg_epochs, fs):
    """
    Description
    -----------
    Function that takes the Fourier Transform of the epoched EEG data and provides the corresponding frequencies.

    Parameters
    ----------
    eeg_epochs : array of floats, size ExCxS, where E is the number of epochs, C is the number of channels, and S is the number of samples within the epoch
        Array containing the EEG data in volts from each of the electrode channels organized by periods of time in which an event (12Hz or 15Hz flashes) occurs.
    fs : array of float, size 1
        The sampling frequency of the data obtained in the 'fs' key of data_dict.

    Returns
    -------
    eeg_epochs_fft : array of complex numbers, size ExCx((fs/2)+1), where E is the number of epochs, C is the number of channels, and fs is the sampling frequency
        The EEG data converted to the frequency space for each epoch and channel.
    fft_frequencies : array of floats, size (fs/2)+1, where fs is the sampling frequency
        Array containing sample frequencies.

    """
  
    # take the Fourier Transform of the epoched EEG data
    eeg_epochs_fft = np.fft.rfft(eeg_epochs)
    
    # find the corresponding frequencies from the epoched EEG data
    fft_frequencies = np.fft.rfftfreq(n=eeg_epochs.shape[-1], d=1/fs) # n is the number of samples in the signal (final dimension) in eeg_epochs), d is the inverse of sampling frequency
  
    return eeg_epochs_fft, fft_frequencies

#%% Part 5: Plot the Power Spectra

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject, is_plotting=True, event_15_normalization_factor=None, event_12_normalization_factor=None):
    """
    Description
    -----------
    Function that uses the Fourier Transform of the epoched EEG data to compute and plot the power spectra of different electrodes.
    
    ***UPDATES***
    Optional input of is_plotting added (default is True) that gives the user the option to suppress the plot when generating the power spectra. Optional inputs of event_15_normalization_factor and event_12_normalization_factor added (defaults are None) that give the user the option to normalize the epoched data to a different dataset. event_15_normalization_factor and event_12_normalization_factor are also outputs and give the factors to which the data were normalized (which may or may not contain the maximum mean powers for the dataset of interest depending on the input).

    Parameters
    ----------
    eeg_epochs_fft : array of complex numbers, size ExCx((fs/2)+1), where E is the number of epochs, C is the number of channels, and fs is the sampling frequency
        The EEG data converted to the frequency space for each epoch and channel.
    fft_frequencies : array of floats, size (fs/2)+1, where fs is the sampling frequency
        Array containing sample frequencies.
    is_trial_15Hz : array of boolean, size Ex1, where E is the number of epochs (or events)
        Array containing True if the epoch is an event at 15Hz, False if the epoch is an event at 12Hz.
    channels : array of str, size Cx1, where C is the number of channels in the dataset (32)
        Array containing strings with the name of each electrode/channel in the dataset.
    channels_to_plot : list, size Cx1, where C is the number of channels to be plotted
        Input containing which channels will be plotted.
    subject : int
        The subject for which the data will be loaded.
    is_plotting : boolean, optional
        A boolean input variable that determines whether the function call will produce a plot. The default is True.
    event_15_normalization_factor : array of floats, size Cx1, where C is the number of channels, optional
        An array containing the maximum power for the 15Hz stimulus at each channel to serve as a normalization factor. The default is None.
    event_12_normalization_factor : array of floats, size Cx1, where C is the number of channels, optional
        An array containing the maximum power for the 12Hz stimulus at each channel to serve as a normalization factor. The default is None.

    Returns
    -------
    spectrum_db_15Hz : array of floats, size Cx((fs/2)+1), where C is the number of channels and fs is the sampling frequency
        Array containing the power spectrum of the 15Hz events for each channel.
    spectrum_db_12Hz : array of floats, size Cx((fs/2)+1), where C is the number of channels and fs is the sampling frequency
        Array containing the power spectrum of the 12Hz events for each channel.
    event_15_normalization_factor : array of floats, size Cx1, where C is the number of channels, optional
        An array containing the maximum power of some dataset for the 15Hz stimulus at each channel that can serve as a normalization factor for other data.
    event_12_normalization_factor : array of floats, size Cx1, where C is the number of channels, optional
        An array containing the maximum power of some dataset for the 12Hz stimulus at each channel that can serve as a normalization factor for other data.

    """

    # convert channels to list
    channels = list(channels)
    
    # calculate power spectra
    # isolate frequency spectra by event type (12Hz or 15Hz)
    event_15 = eeg_epochs_fft[is_trial_15Hz,:,:]
    event_12 = eeg_epochs_fft[~is_trial_15Hz,:,:]
    
    # calculate power for event type
    event_15_power = (np.abs(event_15))**2
    event_12_power = (np.abs(event_12))**2 
    
    # calculate mean power for event type
    event_15_power_mean = event_15_power.mean(0)
    event_12_power_mean = event_12_power.mean(0)
    
    # allow data to be normalized to itself by finding the maximum power in each channel for the data
    if event_15_normalization_factor is None:
    
        # find maximum power by channel
        event_15_normalization_factor = event_15_power_mean.max(1)
    
    if event_12_normalization_factor is None:
        # find maximum power by channel
        event_12_normalization_factor = event_12_power_mean.max(1)
    
    # calculate normalized power for event type
    # preallocate arrays    
    normalized_event_15_power_mean = np.zeros(event_15_power_mean.shape)
    normalized_event_12_power_mean = np.zeros(event_12_power_mean.shape)
    
    # normalize to max (all in a channel) - uses the given input if not None
    for channel_index in range(len(channels)):
        
        normalized_event_15_power_mean[channel_index,:] = event_15_power_mean[channel_index,:]/event_15_normalization_factor[channel_index]
        normalized_event_12_power_mean[channel_index,:] = event_12_power_mean[channel_index,:]/event_12_normalization_factor[channel_index]
    
    # calculate spectra for event type
    spectrum_db_15Hz = 10*(np.log10(normalized_event_15_power_mean))
    spectrum_db_12Hz = 10*(np.log10(normalized_event_12_power_mean))
    
    # plotting
    if is_plotting == True:
        
        # inform user of plotting
        print('\nPlotting frequency data...')
        
        # isolate channel being plotted
        channel_to_plot = [channels.index(channel_name) for channel_name in channels_to_plot]
        
        # set up figure
        figure, channel_plot = plt.subplots(len(channels_to_plot), sharex=True)
        
        for plot_index, channel in enumerate(channel_to_plot): # plot_index to access a subplot
            
            # plot the power spectra by event type
            channel_plot[plot_index].plot(fft_frequencies, spectrum_db_12Hz[channel,:], color='red')
            channel_plot[plot_index].plot(fft_frequencies, spectrum_db_15Hz[channel,:], color='green')
            
            # formatting subplot
            channel_plot[plot_index].set_xlim(0,80)
            channel_plot[plot_index].set_xlabel('frequency (Hz)')
            channel_plot[plot_index].tick_params(labelbottom=True) # shows axis values for each subplot when sharex=True, adapted from Stack Overflow (function and keywords)
            channel_plot[plot_index].set_ylabel('power (dB)')
            channel_plot[plot_index].set_title(f'Channel {channels_to_plot[plot_index]}')
            channel_plot[plot_index].legend(['12Hz','15Hz'], loc='best')
            channel_plot[plot_index].grid()
            
            # plot dotted lines at 12Hz and 15Hz
            channel_plot[plot_index].axvline(12, color='red', linestyle='dotted')
            channel_plot[plot_index].axvline(15, color='green', linestyle='dotted')
        
        # format overall plot
        figure.suptitle(f'SSVEP Subject S{subject} Frequency Content')
        figure.tight_layout()
        
        # save image
        plt.savefig(f'SSVEP_S{subject}_frequency_content.png')
        
        # inform user of plotting completion
        print('Plotting frequency data complete.')
        
    return spectrum_db_15Hz, spectrum_db_12Hz, event_15_normalization_factor, event_12_normalization_factor    