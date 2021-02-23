#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:44:32 2021

@author: beriksso
"""

import sys
sys.path.insert(0, '../functions/')
import tofu_functions as dfs
import pickle

shot_number = 94874

detectors = ['S1_05', 'S2_05']

pulses = {}
time_stamps = {}
for detector in detectors:
    board, channel = dfs.get_board_name(detector)
    pulses[detector] = dfs.get_pulses(shot_number = shot_number, board = board, channel = channel)
    time_stamps[detector] = dfs.get_times(shot_number = shot_number, board = board, channel = channel)
    

with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'wb') as handle:
    pickle.dump(pulses, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open('/common/scratch/beriksso/TOFu/optimization/time_stamps.pickle', 'wb') as handle:
    pickle.dump(time_stamps, handle, protocol = pickle.HIGHEST_PROTOCOL)









