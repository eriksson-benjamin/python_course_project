#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:56:04 2021

@author: beriksso
"""

cimport numpy as cnp
import numpy as np

def subtract_values(cnp.ndarray[cnp.float64_t, ndim = 2] pulse_data, cnp.ndarray[cnp.float64_t, ndim = 2] values):
    cdef int i
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] delta = np.empty([np.shape(pulse_data)[0], np.shape(pulse_data)[1]], dtype = 'float64')
    
    delta = pulse_data - values
#    for i, row in enumerate(pulse_data):
#        delta[i] = row - values[i]

    # Find the index of the first positive value
    return delta <= 0
#    mask = delta <= 0