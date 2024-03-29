#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 08:36:51 2024

filter_ssvep_data.py

This file serves as the module for Lab 4 (Filtering). Included within this file are a series of function definitions that serve to manipulate raw EEG data via filtering and produce corresponding graphs, including those depicting the frequency data of these manipulated signals. The first function, make_bandpass_filter(), uses a finite impulse response bandpass filter of Hanning type to generate an array of filter coefficients to eventually be used on the EEG data and plots the impulse and frequency responses given select frequency data and a filter order. Using the generated coefficients, filter_data() filters the EEG data. The next function, get_envelope(), essentially takes the magnitude of the filtered signal to produce the envelope, which is the plotted against the filtered data. Since get_envelope() can be applied to signals that have undergone filtering using filters with varying frequency data, is is of interest to compare the potential differences in the envelopes (12Hz and 15Hz), which, along with plotting the epochs and their corresponding flash frequency, is the functionality of plot_ssvep_amplitudes(). Finally, plot_filtered_spectra() heavily implements functions generated in Lab 3 to calculate the frequency data for each of the variations of the EEG data: the raw signal, the filtered signal, and the envelope of the filtered signal. This function plots each of these signals in separate subplots for which the power spectra for the 12Hz and 15Hz stimuli are compared for several channels.

Useful abbreviations:
    EEG: electroencephalography
    SSVEP: steady-state visual evoked potentials
    fs: sampling frequency
    FFT: Fast Fourier Transform
    FIR: Finite impulse response
    IIR: Infinite impulse response

@authors: Claire Leahy and Ron Bryant
    
"""

# import packages
from matplotlib import pylab as plt
from scipy.signal import firwin, filtfilt, freqz, hilbert
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum, plot_power_spectrum
import numpy as np

#%% Part 2: Design a Filter

def make_bandpass_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=10, fs=1000):
    '''
    Description
    -----------
    Function to create a finite impulse response bandpass filter of Hanning type. Plots the impulse and frequency responses using the filters.

    Parameters
    ----------
    low_cutoff : float
        The lower frequency to be used in the bandpass filter in Hz.
    high_cutoff : float 
        The higher frequency to be used in the bandpass filter in Hz.
    filter_type : str, optional
        The finite impulse response filter of choice to use in the firwin() function. The default is "hann".
    filter_order : int, optional
        The order of the filter. The default is 10.
    fs : int, optional
        The sampling freqeuncy in Hz. The default is 1000.

    Returns
    -------
    filter_coefficients: array of floats, size (O+1)x1, where O is the filter order
        Numerator coefficients of the finite impulse response filter.

    '''
    
    # get filter coefficients
    nyquist_frequency = fs/2 # get Nyquist frequency to use in filter
    filter_coefficients = firwin(filter_order+1, [low_cutoff/nyquist_frequency, high_cutoff/nyquist_frequency], window='hann', pass_zero='bandpass')

    # get frequency response parameters
    filter_frequencies, frequency_responses = freqz(filter_coefficients, fs=fs)
    frequency_responses_dB=10*(np.log10(frequency_responses*np.conj(frequency_responses))) # use conjugate due to complex numbers

    # create figure
    plt.figure(figsize=(8,6), clear=True) 
    
    # impulse response (subplot 1)
    plt.subplot(2,1,1)
    plt.plot(np.arange(0,len(filter_coefficients))/fs, filter_coefficients)
    # subplot format
    plt.title ('Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Gain')
    plt.grid()
    
    # frequency response (subplot 2)
    plt.subplot(2,1,2)
    plt.plot(filter_frequencies, frequency_responses_dB)
    # subplot format
    plt.xlim(0,40)
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude Gain (dB)')
    plt.grid()
    
    # general figure formatting
    plt.suptitle(f'Bandpass Hann Filter with fc=[{low_cutoff}, {high_cutoff}], order={filter_order}')
    plt.tight_layout()
    
    # save figure
    plt.savefig(f'hann_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}')
    
    return filter_coefficients

#%% Part 3: Filter the EEG Signals

def filter_data(data, b):
    '''
    Description
    -----------
    Function that applies the scipy filtfilt() function to the data using the coefficients of the FIR filter.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    b : Array of floats, size (O+1)x1, where O is the filter order
        Numerator coefficients of the finite impulse response filter.

    Returns
    -------
    filtered_data : 2D array of floats, size CxS, where C is the number of channels and S is the number of samples
        The EEG data, in microvolts, filtered twice.

    '''
    
    # extract data from the dictionary
    eeg = data['eeg']*(10**6) # convert to microvolts
    
    # variables for sizing
    channel_count = len(eeg) # 1st dimension of EEG is number of channels
    sample_count = len(eeg.T) # 2nd dimension of EEG is number of samples
    
    # preallocate array
    filtered_data = np.zeros([channel_count, sample_count])
    
    # apply filter to EEG data for each channel
    for channel_index in range(channel_count):
        
        filtered_data[channel_index,:] = filtfilt(b=b, a=1, x=eeg[channel_index,:])
    
    return filtered_data

#%% Part 4: Calculate the Envelope

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency=None):
    '''
    Description
    -----------
    Given a bandpass filtered group of EEG signals, this function returns the envelope of each, which is reflective of the signal's amplitude at each point. If a channel is selected the data for that channel is graphed.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    filtered_data : array of floats size CxS, where C is the number of channels and S is the number of samples
        The filtered EEG data of each channel.
    channel_to_plot : str, optional
        The channel name for which the data will be plotted. The default is None.
    ssvep_frequency : str, optional
        The frequency of the SSVEP stimulus. The default is None.

    Returns
    -------
    envelope : array of floats size CxS, where C is the number of channels and S is the number of samples
        The evevelope of each bandpass filtered signal in microvolts.
    '''
    
    # extract necessary data from the dictionary
    channels = list(data['channels'])
    fs = data['fs']
    
    # variables for sizing
    channel_count = len(filtered_data) # 1st dimension is number of channels
    sample_count = len(filtered_data.T) # 2nd dimension is number of samples
    
    # preallocate the array
    envelope = np.zeros([channel_count, sample_count])
    
    # get the envelope for each channel
    for channel_index in range(channel_count):
        
        envelope[channel_index]=np.abs(hilbert(x=filtered_data[channel_index]))
        
    # data for title if ssvep_frequency is None
    if ssvep_frequency == None:
        ssvep_frequency = '[Unknown]'
    
    # plot the filtered data and envelope if given a channel to plot 
    if channel_to_plot != None:
        
        # time parameters
        T = filtered_data.shape[1]/fs # total time
        t = np.arange(0,T,1/fs) # time axis to plot
        
        # extract the index of the channel to plot
        channel_index = channels.index(channel_to_plot)
        
        # create figure
        plt.figure(figsize=(8,6), clear=True)
        
        # plotting
        plt.plot(t, filtered_data[channel_index], label='filtered signal')
        plt.plot(t, envelope[channel_index], label='envelope')
        
        # format figure
        plt.title(f'{ssvep_frequency}Hz BPF Data (Channel {channel_to_plot})') 
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)') 
        plt.legend()
        plt.grid()
        
        # save figure
        # plt.savefig(f'{ssvep_frequency}Hz_BPF_data_channel_{channel_to_plot}')

    return envelope  

#%% Part 5: Plot the Amplitudes

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    '''
    Description
    -----------
    Plots the envelope of the two filtered EEG signals together above a plot depicting the epochs by flash frequency.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    envelope_a : array of floats size CxS, where C is the number of channels and S is the number of samples
        The envelope of each bandpass (first frequency) filtered signal in microvolts.
    envelope_b : array of floats size CxS, where C is the number of channels and S is the number of samples
        The envelope of each bandpass (second frequency) filtered signal in microvolts.
    channel_to_plot : str
        The channel name for which the data will be plotted.
    ssvep_freq_a : int
        Corresponds to the frequency of the bandpass filter for envelope_a.
    ssvep_freq_b : int
        Corresponds to the frequency of the bandpass filter for envelope_b.
    subject : int
        Number of the subject of origin for the EEG data.

    Returns
    -------
    None.

    '''
    
    # extract data from the dictionary
    channels = list(data['channels']) # convert to list
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    
    # time parameters
    T = envelope_a.shape[1]/fs # total time, same for envelope_b
    t = np.arange(0,T,1/fs) # time axis

    # extract the index of the channel to plot
    channel_index = channels.index(channel_to_plot)
    
    # find event samples and times
    event_intervals = np.zeros([len(event_samples),2]) # array to contain interval times
    event_ends = event_samples + event_durations # event_samples contains the start samples
    event_intervals[:,0] = event_samples/fs # convert start samples to times
    event_intervals[:,1] = event_ends/fs # convert end samples to times
    
    # initialize figure
    figure, sub_figure = plt.subplots(2, figsize=(8,6), sharex=True)
    
    # top subplot containing flash frequency over span of event
    for event_number, interval in enumerate(event_intervals):
    
        # determine the event frequency to plot (y axis)
        if event_types[event_number] == "12hz":
            event_frequency = 12
    
        else: 
            event_frequency = 15
        
        # plottting the event frequency
        sub_figure[0].hlines(xmin=interval[0], xmax=interval[1], y=event_frequency, color='b') # line
        sub_figure[0].plot([interval[0], interval[1]], [event_frequency,event_frequency], 'bo') # start and end markers
    
    # format top subplot
    sub_figure[0].set_xlabel('Time (s)')
    sub_figure[0].set_ylabel('Flash Frequency')
    sub_figure[0].set_yticks([12,15])
    sub_figure[0].set_yticklabels(['12Hz','15Hz'])
    sub_figure[0].grid()

    # bottom subplot containing envelopes of the filtered signals
    sub_figure[1].plot(t,envelope_a[channel_index,:], label=f'{ssvep_freq_a}Hz Envelope')
    sub_figure[1].plot(t,envelope_b[channel_index,:], label=f'{ssvep_freq_b}Hz Envelope')
    
    # format bottom subplot
    sub_figure[1].set_title('Envelope Comparison')
    sub_figure[1].set_xlabel('Time (s)')
    sub_figure[1].set_ylabel('Voltage (µV)')
    sub_figure[1].grid()
    sub_figure[1].legend()
    
    # format figure
    plt.suptitle(f'Subject {subject} SSVEP Amplitudes')
    plt.tight_layout()
    
    # save figure
    # figure.savefig(f'subject_{subject}_SSVEP_amplitudes_channel_{channel_to_plot}')
    
#%% Part 6: Examine the Spectra

def plot_filtered_spectra(data, filtered_data, envelope, channels=['Fz','Oz'], subject=1, filter_frequency=15):
    '''
    Description
    -----------
    Function to plot the filtered power spectra of raw EEG data, filtered EEG data, and the envelope of that data for electrodes of interest. This code relies upon functions (some of which were modified) from Lab 3.

    Parameters
    ----------
    data : dict, size F, where F is the number of fields (6)
        Data from Python's MNE SSVEP dataset as a dictionary object, where the fields are relevant features of the data.
    filtered_data : array of floats size CxS, where C is the number of channels and S is the number of samples
        The filtered EEG data of each channel.
    envelope : array of floats size CxS, where C is the number of channels and S is the number of samples
        The evevelope of each bandpass filtered signal in microvolts.
    channels : array of str size Cx1 where C is the number of channels, optional
        The channel name for which the data will be plotted. The default is ['Fz','Oz'].
    subject : int, optional
        The subject for which the data will be plotted, used in plot labeling. The default is 1.
    filter_frequency : int, optional
        The frequency of the filter for which the data will be plotted, used in plot labeling. The default is 15.

    Returns
    -------
    None.

    '''
    
    # extract data from the dictionary
    all_channels = list(data['channels']) # convert to list
    fs = data['fs']
    
    # dynamic variables
    channel_count = len(channels)
    
    # power spectra conversions
    # raw data
    raw_epochs, epoch_times, is_trial_15Hz = epoch_ssvep_data(data, eeg_data=None) # epoch data with default conditions
    raw_epochs_fft, fft_frequencies = get_frequency_spectrum(raw_epochs, fs) # frequency spectrum
    raw_spectrum_db_15Hz, raw_spectrum_db_12Hz, raw_event_15_normalization_factor, raw_event_12_normalization_factor = plot_power_spectrum(raw_epochs_fft, fft_frequencies, is_trial_15Hz, all_channels, channels, subject, is_plotting=False, event_15_normalization_factor=None, event_12_normalization_factor=None) # power spectrum
    
    # filtered data
    filtered_epochs = epoch_ssvep_data(data, eeg_data=filtered_data)[0] # epoch filtered data, only need epochs
    filtered_epochs_fft = get_frequency_spectrum(filtered_epochs, fs)[0] # frequency spectrum of filtered data, only need FFT of epochs
    # filtered_spectrum_db_15Hz, filtered_spectrum_db_12Hz = plot_power_spectrum(filtered_epochs_fft, fft_frequencies, is_trial_15Hz, all_channels, channels, subject, is_plotting=False, event_15_normalization_factor=raw_event_15_normalization_factor, event_12_normalization_factor=raw_event_12_normalization_factor)[0:2] # normalized to raw spectra
    filtered_spectrum_db_15Hz, filtered_spectrum_db_12Hz = plot_power_spectrum(filtered_epochs_fft, fft_frequencies, is_trial_15Hz, all_channels, channels, subject, is_plotting=False, event_15_normalization_factor=None, event_12_normalization_factor=None)[0:2] # normalized to itself
    
    # envelope data
    envelope_epochs = epoch_ssvep_data(data, eeg_data=envelope)[0] # epoch envelope data, only need epochs
    envelope_epochs_fft = get_frequency_spectrum(envelope_epochs, fs)[0] # frequency spectrum of envelope data, only need FFT of epochs
    # envelope_spectrum_db_15Hz, envelope_spectrum_db_12Hz = plot_power_spectrum(envelope_epochs_fft, fft_frequencies, is_trial_15Hz, all_channels, channels, subject, is_plotting=False, event_15_normalization_factor=raw_event_15_normalization_factor, event_12_normalization_factor=raw_event_12_normalization_factor)[0:2] # normalized to raw spectra
    envelope_spectrum_db_15Hz, envelope_spectrum_db_12Hz = plot_power_spectrum(envelope_epochs_fft, fft_frequencies, is_trial_15Hz, all_channels, channels, subject, is_plotting=False, event_15_normalization_factor=None, event_12_normalization_factor=None)[0:2] # normalized to itself
    
    # initialize figure
    figure, plots = plt.subplots(channel_count,3, sharex=True, sharey=True, figsize=(12, 8))
    
    for row_index in range(channel_count):
    
        # isolate channel to be plotted
        channel_index = all_channels.index(channels[row_index]) # channel_index in channels_to_plot corresponds to row_index
    
        for column_index in range(3): # 3 columns
            
            # raw data in first column
            if column_index == 0:
                
                plots[row_index][column_index].plot(fft_frequencies, raw_spectrum_db_12Hz[channel_index,:], color='red')
                plots[row_index][column_index].plot(fft_frequencies, raw_spectrum_db_15Hz[channel_index,:], color='green')
                if row_index == 0: # only plot title above first row plots
                    plots[row_index][column_index].set_title('Raw Data Power Spectra')
                plots[row_index][column_index].set_xlabel('Frequency (Hz)')
                plots[row_index][column_index].set_ylabel(f'Channel {all_channels[channel_index]} Power (dB)') # only set y label for first plot (same axis for all plots in row)
            
            # filtered data in second column
            elif column_index == 1:
                
                plots[row_index][column_index].plot(fft_frequencies, filtered_spectrum_db_12Hz[channel_index,:], color='red')
                plots[row_index][column_index].plot(fft_frequencies, filtered_spectrum_db_15Hz[channel_index,:], color='green')
                
                if row_index == 0: # only plot title above first row plots
                    plots[row_index][column_index].set_title('Filtered Data Power Spectra')
                plots[row_index][column_index].set_xlabel('Frequency (Hz)')
            
            # envelope in third column
            elif column_index == 2:
                
                plots[row_index][column_index].plot(fft_frequencies, envelope_spectrum_db_12Hz[channel_index,:], color='red')
                plots[row_index][column_index].plot(fft_frequencies, envelope_spectrum_db_15Hz[channel_index,:], color='green')
                
                if row_index == 0: # only plot title above first row plots
                
                    plots[row_index][column_index].set_title('Envelope Data Power Spectra')
                    plots[row_index][column_index].legend(['12Hz', '15Hz'], title='Stimulus', loc='upper right')
                    
                plots[row_index][column_index].set_xlabel('Frequency (Hz)')
                
            # formatting applied to each subplot
            plots[row_index][column_index].grid()
            plots[row_index][column_index].set_xlim(0,60)
            plots[row_index][column_index].set_ylim(-120,5)
            plots[row_index][column_index].tick_params(labelbottom=True)
            
    # whole figure formatting
    figure.suptitle(f'Subject {subject} Power Spectra with a {filter_frequency}Hz Bandpass Filter')
    figure.tight_layout()
    
    # save figure
    plt.savefig(f'SSVEP_S{subject}_frequency_content_{filter_frequency}Hz_filter.png')
    