#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:11:39 2021

@author: beriksso
"""

cimport numpy as np
import numpy

def CyTOF(np.ndarray[np.float64_t, ndim = 1] S1_times, np.ndarray[np.float64_t, ndim = 1] S2_times, np.float64_t t_back, np.float64_t t_forward, return_indices = False):    
    '''
    We choose an event in S2, define the width of the window in which we want
    to search for an event in S1 and calculate the time difference between 
    the chosen S2 time stamp and the found S1 time stamp(s)
    S1_times: 1D array of time stamps for one S1
    S2_times: 1D array of time stamps for one S2
    t_back: Number of time units (usually ns) to look back in time to find coincidences between S2 and S1 (gives positive TOF's)
    t_forward: Number of time units (usually ns) to look forwards in time to find coincidences between S2 and S1 (gives negative TOF's)
    Example: coincidences = sTOF3(S1_time_stamps, S2_time_stamps, t_back = 400, t_forward = 200)
    '''
    cdef int counter, i, n_S1, n_S2, search_sorted, lowest_index, low_index
    cdef np.ndarray[np.float64_t, ndim = 1] w_low, w_high, dtx,
    cdef np.ndarray[np.int64_t, ndim = 1] ind_S1, ind_S2
    
    
    n_S1 = S1_times.shape[0]
    n_S2 = S2_times.shape[0]
    cdef np.ndarray[np.int64_t, ndim = 2] ind  = numpy.ones([5*n_S2, 2], dtype = 'int64')
    cdef np.ndarray[np.float64_t, ndim = 1] dt = numpy.ones(5*n_S2, dtype = 'float64')

    # Define time windows
    w_low = S2_times - t_back
    w_high = S2_times + t_forward
    
    counter = 0
    finished = False

    lowest_indices = numpy.searchsorted(S1_times, w_low)
    for i in range(0, n_S2):
        search_sorted = 0
        # Find the time stamp in S1 closest to wLow (rounded up, i.e. just outside the window)
        lowest_index = lowest_indices[i]
        while True:
            # Increase to next event
            low_index = lowest_index + search_sorted
            # If the time stamp is the final one in S1 we break
            if lowest_index >= n_S1 - 1 or low_index >= n_S1: 
                finished = True
                break
        
            # If the time stamp in S1 is beyond the window we go to next S2 time (there are no more time stamps within this window)
            if S1_times[low_index] >= w_high[i]: break
            # If the time stamp in S1 is before the window check the next time stamp    
            if S1_times[low_index] <= w_low[i]: 
                search_sorted += 1
                continue
        
            # If there is an event we calculate the time difference
            dt[counter] =  S2_times[i] - S1_times[low_index]
            # Save the S1 and S2 index of the event
            ind[counter][0] = low_index
            ind[counter][1] = i
            counter += 1
            search_sorted += 1
        if finished: break
    
    # Find and remove all fails from dt
    dtx = dt[(dt != -9999)]

    ind_S1 = ind[:, 0][ind[:, 0] != -9999]
    ind_S2 = ind[:, 1][ind[:, 1] != -9999]

    if return_indices:
#        indx[0, :] = ind_S1
#        indx[1, :] = ind_S2
        return dtx, [ind_S1, ind_S2]
    else: return dtx