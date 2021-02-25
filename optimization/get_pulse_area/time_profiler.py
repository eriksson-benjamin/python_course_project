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
from gpa import get_pulse_area

# Load local data
with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'rb') as handle:
    pulses = pickle.load(handle)

with open('/common/scratch/beriksso/TOFu/optimization/time_stamps.pickle', 'rb') as handle:
    time_stamps = pickle.load(handle)

pulses_S1 = pulses['S1_05']
pulses_S2 = pulses['S2_05']

areas_S1 = get_pulse_area(pulses_S1, u_factor = 1)
#areas_S2 = get_pulse_area(pulses_S2, u_factor = 1)



