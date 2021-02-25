#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:35:45 2021

@author: beriksso
"""

import sys
sys.path.insert(0, '../../functions/')
#import tofu_functions as dfs
import pickle
import numpy as np
from tp_CFD import time_pickoff_CFD


def baseline_reduction(pulse_data, timer = False):
    '''
    Returns the same pulse data array with the base line at zero.
    pulse_data: array of pulse height data where each row corresponds to one record.
    '''
#    if timer: t_start = elapsed_time()
    
    # Calculate the average baseline from ten first samples in each record
    baseline = pulse_data[:, :10]
    baseline_av = np.mean(baseline, axis = 1)
    
    # Create array of baseline averages with same size as pulse_data
    baseline_av = np.reshape(baseline_av, (len(baseline_av), 1))
    baseline_av = np.repeat(baseline_av, np.shape(pulse_data)[1], axis = 1)
    
#    if timer: elapsed_time(t_start, 'baseline_reduction()')
    return pulse_data-baseline_av

# Load local data
with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'rb') as handle:
    pulses = pickle.load(handle)

with open('/common/scratch/beriksso/TOFu/optimization/time_stamps.pickle', 'rb') as handle:
    time_stamps = pickle.load(handle)

pulses_S1 = pulses['S1_05']
pulses_S1_bl = baseline_reduction(pulses_S1)
time_of_arrival = time_pickoff_CFD(pulses_S1_bl, fraction = 0.05)



