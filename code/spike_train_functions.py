import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import quantities as pq
from neo.core import SpikeTrain
import neo
import elephant
from elephant.spike_train_dissimilarity import victor_purpura_dist
from joblib import Parallel, delayed
import multiprocessing

def spike_train_trials(sorted_timestamps, units, event, start, stduration, shift_idx=1):
    # Iterate over all units->trials
    st_dict = {}
    for u in range(np.size(units)):
        ts = sorted_timestamps[units[u]]

        event_dict = {}
        for en in range(np.size(event)):
            #Caclulate firing rates and smooth
            win_start = start + event[en] 
            win_end = start + stduration + event[en] 

            spike_train_event = ts[np.logical_and(ts>=win_start,ts<win_end)]
            event_dict[en] = {'ts':spike_train_event, 'win_start':win_start, 'win_end':win_end}
        
        st_dict[( units[u], shift_idx )] = event_dict #Index by


    return st_dict

def spike_train_rates(unit_data, t_start, t_end, fs, kernel, sampling_period=None):
    if sampling_period is None:
        sampling_period == (1/fs)*pq.s

    assert t_end > t_start

    # Imprecision between window times and desired sampling rate may produce length discrepancies
    # This code expands the window by window_buffer, and then trims down to the expected n_samples.
    n_samples = np.round((t_end-t_start) / sampling_period).astype(int)
    window_buffer = 0.1
    
    prefix = np.array([-np.inf, -np.inf]) #Deals with edge case where unit spikes once during trial and returns float 
    unit_data = np.insert(prefix, 2, unit_data) #Transforms to array, -np.inf used to make sure t_start>prefix

    unit_data = unit_data[np.logical_and(unit_data>t_start,unit_data<t_end)]

    unit_train = neo.SpikeTrain(unit_data, units=pq.s,t_stop=t_end + window_buffer, t_start=t_start, sampling_rate=fs*pq.Hz)

    rate = np.array(elephant.statistics.instantaneous_rate(unit_train, sampling_period=sampling_period, kernel=kernel))
    rate = rate[:n_samples]

    return rate

def st_window_split(st_data, event_times, wstart, wstop, shift=True):
    """ Extract windows from continuous data centered at event_times
    Parameters
    ----------
    data : array like of float (2 dimensions)
        Data to be split into windows. Recording channels correspond to rows (dim=0),
        columns (dim=1) correspond to times.
    event_times : array like of float (1 dimension)
        Position where windows are centered.
    wstart : float
        Start time for window centered at event_times.
    wstop : float
        End time for window centered at event_times.
    shift : bool
        Option to substract the window start time for each trial from spike timestamps.
    Returns
    -------
    windowed_units : nested list (3 dimensions)
        (n_units, n_trials, n_spikes)
    """
    assert wstart < wstop

    windowed_units = list()
    for unit_data in st_data:
        windowed_spikes = list()
        for event in event_times:
            times_mask = np.logical_and(unit_data > (event + wstart), unit_data < (event + wstop)).reshape(-1)
            if shift:
                shifted_data = unit_data[times_mask] - (event + wstart)
            else:
                shifted_data = unit_data[times_mask]
            windowed_spikes.append(shifted_data)
        windowed_units.append(windowed_spikes)

    return windowed_units