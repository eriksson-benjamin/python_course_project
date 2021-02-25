#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:35:45 2021

@author: beriksso
"""

import sys
sys.path.insert(0, '../functions/')
import tofu_functions as dfs
import pickle
from sTOF import CyTOF
#import CyTOF
import numpy as np
@profile
def not_cythonized():
    coincidences = dfs.sTOF4(times_S1, times_S2)
    
@profile
def cythonized():
        coincidences = CyTOF.CyTOF(times_S1, times_S2, t_forward = 100, t_back = 100)

# Load local data
with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'rb') as handle:
    pulses = pickle.load(handle)

with open('/common/scratch/beriksso/TOFu/optimization/time_stamps.pickle', 'rb') as handle:
    time_stamps = pickle.load(handle)

times_S1 = time_stamps['S1_05']
times_S2 = time_stamps['S2_05']

# Time profiling of sTOF() function
coincidences = not_cythonized()
C_coincidences = cythonized()

# Check that results are the same
print(f'coincidences == C_coincidences: {np.all(coincidences == C_coincidences)}')


