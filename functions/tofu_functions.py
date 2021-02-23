#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:27:51 2019

@author: beriksso
"""

import getdat as gd
import ppf
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.constants as constant
import scipy.optimize as optimize
from matplotlib.lines import Line2D

def get_pulses(board, channel, shot_number, pulse_start = -1, pulse_end = -1, timer = False):
    '''
    Returns pulse data for a given board, channel and shot number.
    Example how to call: data = get_pulses('04', 'A', 94206) 
    '''
    if timer: t_start = elapsed_time()
    file_name = 'M11D-B' + board + '<DT' + channel
    
    # Record length is 64 for ADQ14 and 56 for ADQ412
    if   board in ('01', '02', '03', '04', '05'): record_length = 64
    elif board in ('06', '07', '08', '09', '10'): record_length = 56
    else: 
        raise Exception('Wrong function call. Example call: data = get_pulses(\'04\', \'A\', 94206)')
    
    # Get some of the data or all of it
    if (pulse_start != -1) & (pulse_end != -1):
        pulse_data, nD, ier = gd.getbytex(file_name, shot_number, nbytes = (pulse_end - pulse_start) * record_length * 2, start = 6 + 2 * record_length * pulse_start, order = 12)
    else:
        pulse_data, nD, ier = gd.getbyte(file_name, shot_number)
    pulse_data.dtype = np.int16

  
#    # Get pulse data
#    pulse_data = gd.getbyte(file_name, shot_number)[0]
#    pulse_data.dtype = np.int16
#    
    # Reshape pulse data
    if len(pulse_data) % record_length != 0: 
        raise Exception('Error: Number of records could not be calculated for record length of 64 samples.')

    if timer: elapsed_time(t_start, 'get_pulses()')
    return np.reshape(pulse_data, [int(len(pulse_data) / record_length), record_length])
       
def get_times(shot_number, board = 'N/A', channel = 'N/A', detector_name = 'N/A', timer = False):
    '''
    Returns time stamps for given board, channel and shot number in nanoseconds.
    Example how to call: times = get_times('02', 'B', 94206)
    '''
    if timer: t_start = elapsed_time()
    if detector_name != 'N/A': board, channel = get_board_name(detector_name)
    file_name = 'M11D-B' + board + '<TM' + channel

    # For ADQ14 time stamps are multiplied by 0.125 to return in ns
    # For ADQ412 time stamps are multiplied by 0.5 to return in ns
    if   board in ('01', '02', '03', '04', '05'): mult_factor = 0.125
    elif board in ('06', '07', '08', '09', '10'): mult_factor = 0.5
    else:
        print('Wrong function call. Example call: data = get_pulses(\'04\', \'A\', 94206)')
        return 0
    
    # Get time stamps
    time_stamps = gd.getbyte(file_name, shot_number)[0]
    if time_stamps.size == 0: return -1
    time_stamps.dtype = np.uint64
    
    if timer: elapsed_time(t_start, 'get_times()')
    return time_stamps * mult_factor

def get_offset(board, shot_number, timer = False):
    '''
    Returns offsets for given ADQ412 board, channel and shot number.
    Example how to call: offsets = get_offset('07', 94206)
    '''
    if timer: t_start = elapsed_time()
    file_name = 'M11D-B' + board + '<OFF' 
    if board not in ('06', '07', '08', '09', '10'):
        print('Offsets are only available for the ADQ412 cards (i.e. boards 06, 07, 08, 09 and 10)')
        return 0
    
    # Get offset
    offset = gd.getbyte(file_name, shot_number)[0]
    offset.dtype = np.uint64
    
    if timer: elapsed_time(t_start, 'get_offset()')
    if len(offset) == 0: 
        print('get_offset() failed to retrieve offset value.')
        return [] 
    return offset[0]
    
def get_temperatures(board, shot_number, timer = False):
    '''
    Returns temperatures (deg C) at different locations on the ADQ412 and ADQ14 boards before and after acquisition.
    For ADQ412 five temperatures are returned, two of which are always 256, these temperature locations on the boards
    are not available for our cards. For ADQ14 seven temperatures are returned.
    For information on where the temperatures are measured see the function GetTemperature() in the ADQAPI manual.
    Example how to call: T_before, T_after = get_temperatures('04', 94207)
    '''
    if timer: t_start = elapsed_time()
    
    file_name_1 = 'M11D-B' + board + '<T0'
    file_name_2 = 'M11D-B' + board + '<TE'
    
    # Get temperatures
    T0 = gd.getbyte(file_name_1, shot_number)[0]
    TE = gd.getbyte(file_name_2, shot_number)[0]
    T0.dtype = np.float32
    TE.dtype = np.float32
    
    if timer: elapsed_time(t_start, 'get_temperatures()')
    return T0, TE
    
def get_trigger_level(board, channel, shot_number, timer = False):
    '''
    Returns the trigger level used for the given shot, board and channel
    Example how to call: trig_level = get_trigger_level('01', 'A', 94207)
    '''
    if timer: t_start = elapsed_time()
    
    file_name = 'M11D-B' + board + '>TL' + channel
    
    # Get trigger level
    tlvl, nD, ier = gd.getbyte(file_name, shot_number)
    tlvl.dtype = np.int16
    
    if timer: elapsed_time(t_start, 'get_trigger_level()')
    return tlvl.byteswap()[0]

def get_pre_trigger(board, shot_number, timer = False):
    '''
    Returns the number of pre trigger samples used for the given shot and board.
    Example how to call: trig_level = get_trigger_level('01', 'A', 94207)
    '''
    if timer: t_start = elapsed_time()
    
    file_name = 'M11D-B' + board + '>PRT'
    
    # Get number of pre trigger samples
    prt, nD, ier = gd.getbyte(file_name, shot_number)
    
    
    if timer: elapsed_time(t_start, 'get_pre_trigger()')
    return prt[1]

def get_bias_level(board, shot_number, timer = False):
    '''
    Returns the bias level for the given shot, board and channel.
    Example how to call: bias_level = get_bias_level('01', 'A', 94207)
    '''    
    if timer: t_start = elapsed_time()
    file_name = 'M11D-B' + board + '>BSL'
    
    # Get bias level
    blvl, nD, ier = gd.getbyte(file_name, shot_number)
    blvl.dtype = np.int16    
    
    if timer: elapsed_time(t_start, 'get_bias_level()')
    return blvl.byteswap()[0]

def baseline_reduction(pulse_data, timer = False):
    '''
    Returns the same pulse data array with the base line at zero.
    pulse_data: array of pulse height data where each row corresponds to one record.
    '''
    if timer: t_start = elapsed_time()
    
    # Calculate the average baseline from ten first samples in each record
    baseline = pulse_data[:, :10]
    baseline_av = np.mean(baseline, axis = 1)
    
    # Create array of baseline averages with same size as pulse_data
    baseline_av = np.reshape(baseline_av, (len(baseline_av), 1))
    baseline_av = np.repeat(baseline_av, np.shape(pulse_data)[1], axis = 1)
    
    if timer: elapsed_time(t_start, 'baseline_reduction()')
    return pulse_data-baseline_av

def remove_led(time_stamps, timer = False):
    '''
    Returns the same time stamps back without the LED chunk at the end and the position where the LED starts.
    time_stamps: one dimensional array of time stamps
    '''
    if timer: t_start = elapsed_time()
    
    if len(time_stamps) == 0:
        print('Time stamp array is empty.')
        return 0, 0
    
    # Find time difference between each time stamp
    dT = np.diff(time_stamps)
   
    dT_arg = np.where((dT > 190000) & (dT < 200000))[0]
    
    combo_counter = 0
    A = 0
    for led_start in dT_arg:
        
        B = led_start - A
        if B == 1: combo_counter += 1
        else: combo_counter = 0
        A = led_start    
        
        if combo_counter == 10: break
    
    if combo_counter < 10: print('LED\'s not found, returning full data set.')
    
    if timer: elapsed_time(t_start, 'remove_led()')
    return time_stamps[0:led_start - 10], led_start - 10

def find_threshold(pulse_data, trig_level, timer = False, detector_name = 'None'):
    '''
    Finds the point in the pulse which crosses the trigger level (generally 16, 17, 18 or 19 ns for ADQ14).
    Mainly relevant for ADQ14 cards since the number of pre trigger samples varies.
    pulse_data: array of pulse height data where each row corresponds to one record.
    trig_level: trigger level used during acquisition of the data set
    '''
    if timer: t_start = elapsed_time()
    # Subtract the trigger level from pulse data
    pulse_data = pulse_data - trig_level

    # Find all negative numbers (positive numbers correspond to elements above the threshold)
    neg_pulse_data = np.where(pulse_data <= 0)

    # Find the index of the first ocurrence of each number in neg_pulse_data
    # Example: neg_pulse_data[0] = [0(this one), 0, 0, 0, 0, 1(this one), 1, 1, 2(this one), 2, 2, 2...]
    u, indices = np.unique(neg_pulse_data[0], return_index = True)

    # Choose the corresponding elements from neg_pulse_data[1]
    thr_crossing = neg_pulse_data[1][indices]
    if timer: elapsed_time(t_start, 'find_threshold()')

    return thr_crossing

def sinc_interpolation(pulse_data, x_values, ux_values, timer = False):
    '''
    Performs sinc interpolation on the pulse data set.
    pulse_data: array of pulse height data where each row corresponds to one record. 
                NOTE: pulse_data must be baseline reduced (see baseline_reduction() function), otherwise sinc interpolation fails.
    x_values:   one dimensional array of values corresponding to pulse_data's x-axis.
                must have a constant period.
    ux_values:  one dimensional array similar to x_values but upsampled.
    Example:
        pulse_data is the regular m*n array where m = number of records and n = number of samples per record
        x_axis = np.arange(0, n)
        u_factor = 10 is the upsampling factor
        ux_axis = np.arange(0, n, 1./u_factor)
        u_pulse_data = sinc_interpolation(pulse_data, x_axis, ux_axis)
    From Matlab example: http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    '''
    if timer: t_start = elapsed_time()
    
    # Record length
    length = np.shape(pulse_data)[1]
    n_records = len(pulse_data)
    
    # Store results here
    u_pulse_data = np.zeros([len(pulse_data), len(ux_values)])
    counter = 0
    
    if length != len(x_values):
        print('pulse_data and x_values must be the same length.')
        return 0
    
    # Chunk data if too many records
    if n_records > 1E+6:
        # Chunk array in chunks of ~1E6 rows
        n_chunks = int(np.ceil(len(pulse_data) / 1E+6))
        chunked_data = np.array_split(pulse_data, n_chunks, axis = 0)
    # Otherwise use full data set at once
    else: chunked_data = [pulse_data]
    
    # Do sinc interpolation for each chunk of data
    for pulse_data in chunked_data:
        # Find period of x_values
        period = x_values[1] - x_values[0]    
        
        # Set up sinc matrix
        sinc_matrix = np.tile(ux_values, (len(x_values), 1)) - np.tile(x_values[:, np.newaxis], (1, len(ux_values)))
        
        # Perform sinc interpolation
        sinc = np.sinc(sinc_matrix / period)
        u_pulse_data[counter:len(pulse_data) + counter, :] = np.dot(pulse_data, sinc)
        counter += len(pulse_data)
    
    if timer: elapsed_time(t_start, 'sinc_interpolation()')
    return u_pulse_data
    
#def time_pickoff_CFD(pulse_data, fraction = 0.3, timer = False):
#    '''
#    
#    '''
#    if timer: t_start = elapsed_time()
#    
#    # Find the minima and a fraction of the minima
#    minima = np.min(pulse_data, axis = 1)
#    minima_fraction = minima * fraction
#    # Find position of minimum
##    minima_pos = np.argmin(pulse_data, axis = 1)
##    print('Warning: ' + str(len(minima_pos[minima_pos < 100])) + ' pulses have minimum before 10 ns.')
#    
#
#    # Find the index of the point closest to the fraction of the minimum
#    # Look only in the first 25 ns (leading edge) of the pulse
#    x_closest = find_points(pulse_data[:, 0:250], minima_fraction, timer = timer)
#
#
#    # Set up for simple linear regression
#    reg_x = np.zeros([len(x_closest), 3])
#    reg_y = np.zeros([len(x_closest), 3])
#    array_1D = np.arange(0, len(pulse_data), 1)
#    
#    # Choose the three points on which to perform simple linear regression
#    reg_y[:, 0] = pulse_data[array_1D, x_closest - 1]
#    reg_y[:, 1] = pulse_data[array_1D, x_closest]
#    reg_y[:, 2] = pulse_data[array_1D, x_closest + 1]
#
##    reg_y[:, 0] = pulse_data[array_1D, x_closest - 2]
##    reg_y[:, 1] = pulse_data[array_1D, x_closest - 1]
##    reg_y[:, 2] = pulse_data[array_1D, x_closest]
#
#    reg_x[:, 0] = x_closest - 1
#    reg_x[:, 1] = x_closest
#    reg_x[:, 2] = x_closest + 1
#    
#    # Perform simple linear regression
#    slopes, intercepts = linear_regression(reg_x, reg_y, timer = timer)
#    
#    # Solve the y = kx + m equation for x. y = minima_fraction
#    new_time = (minima_fraction - intercepts) / slopes
#    
#    if timer: elapsed_time(t_start, 'time_pickoff_CFD()')
#    return new_time

def time_pickoff_CFD(pulse_data, fraction = 0.3, timer = False):
    '''
    Returns the times of arrival for a 2D array of pulses using a constant
    fraction + linear interpolation method.
    pulse_data: 2D array of pulses where each row corresponds to one pulse
    fraction: fraction at which to perform linear interpolation
    return a 1D array of times-of-arrival for each pulse.
    '''
    
    new_time = np.zeros([len(pulse_data)])

    # Determine whether data needs to be chunked or not
    if len(pulse_data) > 5E+5: chunk_data = True
    else: chunk_data = False

    if timer: t_start = elapsed_time()
    if chunk_data:
        # Chunk array in chunks of ~5E5 rows
        n_chunks = int(np.ceil(len(pulse_data) / 5E+5))
        chunked_data = np.array_split(pulse_data, n_chunks, axis = 0)
        

    else: chunked_data = [pulse_data]
    new_time_counter = 0
    for pulse_data in chunked_data:
        # Find the minima and a fraction of the minima
        minima = np.min(pulse_data, axis = 1)
        minima_fraction = minima * fraction
        # Find position of minimum
    #    minima_pos = np.argmin(pulse_data, axis = 1)
    #    print('Warning: ' + str(len(minima_pos[minima_pos < 100])) + ' pulses have minimum before 10 ns.')
        
    
        # Find the index of the point closest to the fraction of the minimum
        # Look only in the first 25 ns (leading edge) of the pulse
        x_closest = find_points(pulse_data[:, 0:250], minima_fraction, timer = timer)
    
    
        # Set up for simple linear regression
        reg_x = np.zeros([len(x_closest), 3])
        reg_y = np.zeros([len(x_closest), 3])
        array_1D = np.arange(0, len(pulse_data), 1)
        
        # Choose the three points on which to perform simple linear regression
        reg_y[:, 0] = pulse_data[array_1D, x_closest - 1]
        reg_y[:, 1] = pulse_data[array_1D, x_closest]
        reg_y[:, 2] = pulse_data[array_1D, x_closest + 1]
    
        reg_x[:, 0] = x_closest - 1
        reg_x[:, 1] = x_closest
        reg_x[:, 2] = x_closest + 1
        
        # Perform simple linear regression
        slopes, intercepts = linear_regression(reg_x, reg_y, timer = timer)
        # Solve the y = kx + m equation for x. y = minima_fraction
        new_time[new_time_counter:len(pulse_data)+new_time_counter] = (minima_fraction - intercepts) / slopes
        new_time_counter += len(pulse_data)
        

    if timer: elapsed_time(t_start, 'time_pickoff_CFD()')
    return new_time


 
def linear_regression(x_data, y_data, timer = False):
    '''
    Returns the slope (A) and intersection (B) for a simple linear regression on x and y data.
    x_data: 2D array of values where each row corresponds to one event to perform linear regression on
    y_data: 2D array of values where each row corresponds to one event to perform linear regression on
    product_1, product_2 and product_3 correspond to the three products for calculating beta in 
    https://en.wikipedia.org/wiki/Simple_linear_regression
    '''
    if timer: t_start = elapsed_time()
    
    # Find average
    x_mean = np.mean(x_data, axis = 1)
    y_mean = np.mean(y_data, axis = 1)
    
    product_1 = np.transpose(np.transpose(x_data) - x_mean)
    product_2 = np.transpose(np.transpose(y_data) - y_mean)
    product_3 = product_1 ** 2
    
    # Calculate slopes and intersection (y = slope*x + intercept)
    slope = np.sum(product_1 * product_2, axis = 1) / np.sum(product_3, axis = 1)
    intercept = np.mean(y_data, axis = 1) - slope * x_mean
    
    if timer: elapsed_time(t_start, 'linear_regression()')    
    return slope, intercept


    

def find_points(pulse_data, value, timer = False):
    '''
    Returns the index of the point closest to "value" in pulse_data.
    pulse_data: array of pulse height data where each row corresponds to one record. 
                NOTE: pulse_data must be baseline reduced (see baseline_reduction() function).
    value: one dimensional array of values for which you want to find the closest index in pulse_data 
    '''
    if timer: t_start = elapsed_time()
    
    # Subtract the constant fraction value from the data set
    delta = pulse_data - value[:, None]
    
    # Find the index of the first positive value
    mask = delta <= 0
    
    index = np.argmax(mask, axis = 1) 
    
    if timer: elapsed_time(t_start, 'find_points()')
    return index   
    

def sTOF4(S1_times, S2_times, t_back = 100, t_forward = 100, return_indices = False, timer = False):

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
    if timer: t_start = elapsed_time()
    # Define time windows
    w_low = S2_times - t_back
    w_high = S2_times + t_forward
    
    # We will store time differences in dt
    dt = -9999 * np.ones(5 * len(S2_times))
    ind = -9999 * np.ones([5 * len(S2_times), 2])
    counter = 0
    finished = False
    
    for i in range(0, len(S2_times)):
        
        search_sorted = 0
        # Find the time stamp in S1 closest to wLow (rounded up, i.e. just outside the window)
        lowest_index = np.searchsorted(S1_times, w_low[i])
        while True:
            # Increase to next event
            low_index = lowest_index + search_sorted
            # If the time stamp is the final one in S1 we break
            if lowest_index >= len(S1_times) - 1 or low_index >= len(S1_times): 
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

    if timer: elapsed_time(t_start, 'sTOF4()')
    if return_indices:
        indx = np.array([ind_S1, ind_S2], dtype = 'int')
        return dtx, indx
    else: return dtx
    
 

def get_detector_name(board, channel, timer = False):
    '''
    Returns the detector name corresponding to the given board and channel.
    The order of the S2's on the ADQ412's are sadly back to front :(
    Example: detector = get_detector_name('02', 'A')
    '''
    if timer: t_start = elapsed_time()
    detectors = ['S1_01', 'S2_01', 'S2_02', 'S2_03', 
                 'S1_02', 'S2_04', 'S2_05', 'S2_06',
                 'S1_03', 'S2_07', 'S2_08', 'S2_09',
                 'S1_04', 'S2_10', 'S2_11', 'S2_12',
                 'S1_05', 'S2_13', 'S2_14', 'S2_15',
                 'S2_31', 'S2_32', 'ABS_REF', '1kHz_CLK',
                 'S2_27', 'S2_28', 'S2_29', 'S2_30',
                 'S2_23', 'S2_24', 'S2_25', 'S2_26', 
                 'S2_19', 'S2_20', 'S2_21', 'S2_22',
                 'DEAD', 'S2_16', 'S2_17', 'S2_18']
    cha = np.array(['A', 'B', 'C', 'D'])

    if timer: elapsed_time(t_start, 'get_detector_name()')    
    return detectors[4 * (int(board)-1) + np.where(channel == cha)[0][0]]    

def get_board_name(detector_name, timer = False):
    '''
    Returns the board and channel for the given detector name.
    Example: board, channel = get_board_name(detector_name = 'S1_04') returns board = '04', channel = 'A'
    '''
    if timer: t_start = elapsed_time()

    detectors = np.array(['S1_01', 'S2_01', 'S2_02', 'S2_03', 
                 'S1_02', 'S2_04', 'S2_05', 'S2_06',
                 'S1_03', 'S2_07', 'S2_08', 'S2_09',
                 'S1_04', 'S2_10', 'S2_11', 'S2_12',
                 'S1_05', 'S2_13', 'S2_14', 'S2_15',
                 'S2_31', 'S2_32', 'ABS_REF', '1kHz_CLK',
                 'S2_27', 'S2_28', 'S2_29', 'S2_30',
                 'S2_23', 'S2_24', 'S2_25', 'S2_26', 
                 'S2_19', 'S2_20', 'S2_21', 'S2_22',
                 'DEAD', 'S2_16', 'S2_17', 'S2_18'])
    
    channels = np.array(['A', 'B', 'C', 'D'])
    pos = np.where(detectors == detector_name)[0]

    # Find board number
    board = int((np.floor(pos / 4))[0] + 1)
    # Add '0' in front
    if board < 10: board = '0' + str(board)
    else: board = str(board)
    
    # Find channel
    cha = channels[pos % 4][0]
    if timer: elapsed_time(t_start, 'get_detector_name()')    
    return board, cha

def get_shifts(shift_file, timer = False):
    '''
    Returns the shifts (written in shift.txt) required to line up all S1-S2 combinations.
    '''
    if timer: t_start = elapsed_time()
    A = np.loadtxt(shift_file, dtype = 'str')
    
    # Get gamma peak shifts for S1-5 vs S2's
    gamma_peak = np.array(A[0:-4, 1], dtype = 'float')
    # Get neutron peak shifts for S1-5 vs S1's
    neutron_peak = np.array(A[-4:, 1], dtype = 'float')
     
    # Gamma peak should be located at 3.7 ns
    g_peak = 3.7
    
    # Dictionary
    shifts = {'S1_01':[], 'S1_02':[], 'S1_03':[], 'S1_04': [], 'S1_05':[]}

    # This gives how much one needs to shift each TOF spectrum in order to line up with the S1_5 vs S2's at 4 ns
    shifts['S1_05'] = g_peak - gamma_peak
    shifts['S1_04'] = shifts['S1_05'] - neutron_peak[3]
    shifts['S1_03'] = shifts['S1_05'] - neutron_peak[2]
    shifts['S1_02'] = shifts['S1_05'] - neutron_peak[1]
    shifts['S1_01'] = shifts['S1_05'] - neutron_peak[0]
    
    if timer: elapsed_time(t_start, 'get_shifts()')
    return shifts

def get_pulse_area(pulses, u_factor, timer = False):
    '''
    Returns the areas under an array of pulses
    pulses: m*n array of pulses
    u_factor: frequency of samples in each pulse (u_factor = 10 -> 1/10 ns between each sample)
    '''
    
    if timer: t_start = elapsed_time()
    
    # Chunk data if too many pulses
    pulse_area = np.zeros(len(pulses))
    
    if len(pulses) > 1E+6:
        # Chunk array in chunks of ~1E6 rows
        n_chunks = int(np.ceil(len(pulses) / 1E+6))
        chunked_data = np.array_split(pulses, n_chunks, axis = 0)
    
    # Otherwise use full data set at once
    else: chunked_data = [pulses]
    
    # Find area under pulse
    counter = 0
    for chunk in chunked_data:
        pulse_area[counter:len(chunk) + counter] = np.trapz(chunk, axis = 1, dx = 1. / u_factor)
        counter += len(chunk)
        
    if timer: elapsed_time(t_start, 'get_pulse_area()')
    return pulse_area

def get_energy_calibration(areas, detector_name, timer = False):
    '''
    Takes an array of baseline reduced pulse areas and the detector type (S1_01 to S1_05 or S2_01 to S2_32) 
    and returns an array of corresponding deposited energy using the energy
    calibration given in energy_calibration_S1.txt and energy_calibration_S2.txt.
    Example: energy_array = get_energy_calibration(areas, detector_name = 'S1_04')
    '''
    if timer: t_start = elapsed_time()
    
    raise_exception = False
    # Load calibration data for given detector
    if detector_name[0:2] == 'S1':
        cal = np.loadtxt('energy_calibration/energy_calibration_S1.txt', usecols = (0,1))[int(detector_name[3:]) - 1]
        cal_factor = 3000.
    elif detector_name[0:2] == 'S2':
        cal = np.loadtxt('energy_calibration/energy_calibration_S2.txt', usecols = (0,1))[int(detector_name[3:]) - 1]
        if int(detector_name[3:]) <= 15: cal_factor = 3000.
        elif int(detector_name[3:]) > 15: cal_factor = 350.
        else: raise_exception = True   
    else: raise_exception = True
    if raise_exception: raise Exception('Please supply the detector type as the second parameter (SX = \'S1_x\' x = [01, 05] or SX = \'S2_x\' x = [01, 32])')        
    
    # Calculate energy from area
    energy_array = (cal[0] + cal[1] * areas / cal_factor ) / 1000.
    if timer: elapsed_time(t_start, 'get_energy_calibration()')
    return energy_array

def find_time_range(shot_number):
    '''
    Returns the time range for the analysis by using fission chamber data and
    calculating the time at which 99.5% of all neutron events have ocurred.
    Assumes shot start at 40 seconds.
    '''
    
    # Import fission chamber information
    f_chamber = ppf.ppfget(shot_number, dda = "TIN", dtyp = "RNT")
    f_data = f_chamber[2]
    f_times = f_chamber[4]

    if len(f_times) == 0: 
        print('WARNING: Fission chamber data unaviailable.')
        time_slice = np.array([40., 70.])
    else:
        # Create cumulative distribution function
        f_cdist = np.cumsum(f_data) / np.sum(f_data)
        
        # Find the time before which 99.5% of the neutron yield occurs
        time_slice = np.array([40., f_times[np.searchsorted(f_cdist, 0.995)]])
 
    return time_slice

def cleanup(pulses, dx, detector_name, bias_level, baseline_cut = np.array([0, 0]), timer = False):
    '''
    Takes an array of baseline reduced pulses and removes junk pulses.
    pulses: array of baseline reduced pulses
    dx: distance between each point on the x-axis
    detector_name: string containing the name of the detector ('S1_01', ..., 'S2_32')
    bias_level: value in codes where the baseline is expected (typically 27000 for ADQ14, 1600 for ADQ412)
    Example: new_pulses, junk_indices = cleanup(pulses)
    '''
    if timer: t_start = elapsed_time()
    
    # Remove anything with a negative area
    area = np.trapz(pulses, axis = 1, dx = dx)
    indices = np.where(area < 0)[0]

    # Remove anything with points on the baseline far from requested baseline 
    if bias_level not in [27000, 30000, 1600]: print('WARNING: The function cleanup() bases it\'s cuts on a bias level of 27k or 30k for ADQ14 and 1.6k codes for ADQ412. This shot has a bias level of ' + str(bias_level) + ' codes.')
    
    # Define ADQ14 and ADQ412 thresholds for the baseline
    if not np.array_equal(baseline_cut, np.array([0, 0])):
        low_threshold = baseline_cut[0]
        high_threshold = baseline_cut[1]
    elif int(detector_name[3:]) < 16:
        high_threshold = 200
        low_threshold = -200
    elif int(detector_name[3:]) >= 16:
        high_threshold = 70
        low_threshold = 20
    else: raise Exception('Unknown detector name.')
    
    # Find baselines which violate the thresholds
    baseline = pulses[:, 0:10]
    odd_bl = np.unique(np.where((baseline < low_threshold) | (baseline > high_threshold))[0])
    
    # Add to indices, remove duplicates and sort in ascending manner
    indices = np.sort(np.unique(np.append(indices, odd_bl)))
    
    
    if timer: elapsed_time(t_start, 'cleanup()')
    return pulses[indices], indices

def inverted_light_yield(light_yield, timer = False):
    '''
    Takes array of light yields in MeVee and translates to proton recoil using
    look-up table of the inverted light yield function from M. Gatu Johnson.
    '''
    if timer: t_start = elapsed_time()
    # Import look-up table
    table = np.loadtxt('inverted_light_yield/look_up/look_up.txt')
    
    # Find closest value in look-up table for the light yield
    proton_recoil = np.zeros(np.shape(light_yield))
    for i, ly in enumerate(light_yield): 
        arg = np.searchsorted(table[:, 1], ly)
        proton_recoil[i] = table[arg][0]
    
    if timer: elapsed_time(t_start, 'inverted_light_yield()')
    return proton_recoil
    
def light_yield_function(energy, timer = False):
    '''
    Takes array of proton recoil energies in MeV and translates to light yield
    in MeVee using light yield function from M. Gatu Johnson thesis.
    '''
    if timer: t_start = elapsed_time()
    # Different energy ranges
    low_mask    = energy <= 1.9
    medium_mask = (energy > 1.9) & (energy <= 9.3)
    high_mask   = energy > 9.3
    
    a1 = 0.0469
    b1 = 0.1378
    c1 = -0.0183
    
    a2 = -0.01420
    b2 =  0.12920
    c2 =  0.06970
    d2 = -0.00315
    
    a3 = -1.8899
    b3 = 0.7067

    light_yield = np.zeros(np.shape(energy))

    light_yield[low_mask] = (
                             a1 * energy[low_mask]    + 
                             b1 * energy[low_mask]**2 + 
                             c1 * energy[low_mask]**3
                             )

    light_yield[medium_mask] = (
                                a2 + 
                                b2 * energy[medium_mask] +  
                                c2 * energy[medium_mask]**2 +
                                d2 * energy[medium_mask]**3
                                )

    light_yield[high_mask] = (
                              a3 + 
                              b3 * energy[high_mask]
                              )
    if timer: elapsed_time(t_start)
    return light_yield



def get_kincut_function(tof, timer = False):
    '''
    Takes an array of times of flight [ns] and returns the corresponding maximal/minimal
    light yield for each time of flight in MeVee. 
    Input: 
        tof: 1D array of times of flight [ns]
    Output:
        E_S1_max: maximal energy in S1 for given time of flight [MeVee]
        E_S1_min: minimal energy in S1 for given time of flight [Mevee]
        E_S2_max: maximal energy in S2 for given time of flight [MeVee]
    '''
    
    if timer: t_start = elapsed_time()

    l_S2    = 0.35            # length of S2 [m]
    phi_max = np.deg2rad(115) # obtuse angle between S2 and line from S1 centre to S2 centre
    phi_min = np.deg2rad(65)  # acute  angle between S2 and line from S1 centre to S2 centre
    alpha   = np.deg2rad(30)  # angle between S1 and S2 centre w.r.t. centre of constant time-of-flight sphere
    r       = 0.7046          # time-of-flight sphere radius [m]
    
    # Calculate length between S1 and S2 centres [m]
    l = r * np.sin(np.pi - 2*alpha) / np.sin(alpha)

    # Maximum and minimum distances
    l_max = np.sqrt(l**2 + (l_S2/2)**2 - l*l_S2*np.cos(phi_max))
    l_min = np.sqrt(l**2 + (l_S2/2)**2 - l*l_S2*np.cos(phi_min))

    # Maximum and minimum scattering angles
    alpha_max = alpha + np.arccos((l**2 + l_min**2 - l_S2**2/4) / (2 * l * l_min))
    alpha_min = alpha - np.arccos((l**2 + l_max**2 - l_S2**2/4) / (2 * l * l_max))
    
    J_to_MeV = 1E-6 / constant.electron_volt 
#    E_S1_max = 0.5 * constant.neutron_mass * (l_max / (tof*1E-9))**2 * (1 / np.cos(alpha_max)**2 - 1) * J_to_MeV
#    E_S1_min = 0.5 * constant.neutron_mass * (l_min / (tof*1E-9))**2 * (1 / np.cos(alpha_min)**2 - 1) * J_to_MeV
#    E_S2_max = 0.5 * constant.neutron_mass * (l_max / (tof*1E-9))**2 * J_to_MeV

    E_S1_max = 0.5 * constant.neutron_mass * (l_min / (tof*1E-9))**2 * (1 / np.cos(alpha_max)**2 - 1) * J_to_MeV
    E_S1_min = 0.5 * constant.neutron_mass * (l_max / (tof*1E-9))**2 * (1 / np.cos(alpha_min)**2 - 1) * J_to_MeV
    E_S2_max = 0.5 * constant.neutron_mass * (l_max / (tof*1E-9))**2 * J_to_MeV


    # Translate to light yield
    ly_S1_max = light_yield_function(E_S1_max)
    ly_S1_min = light_yield_function(E_S1_min)
    ly_S2_max = light_yield_function(E_S2_max)

    if timer: elapsed_time(t_start, 'get_kincut_function()')
    return ly_S1_min, ly_S1_max, ly_S2_max

def kinematic_cuts(tof, energy_S1, energy_S2, timer = False):
    '''
    Performs kinematic cuts on the times of flight vs. energy for S1's and S2's.
    Input:
        tof:       array of times of flight [ns]
        energy_S1: array of S1 energies [MeVee]
        energy_S2: array of S2 energies [MeVee]
    Output:
        tof_cut:       1D array of times of flight with kinematic cuts applied
        energy_S1_cut: 1D array of energies for S1 with kinematic cuts applied
        energy_S2_cut: 1D array of energies for S2 with kinematic cuts applied
    '''
    if timer: t_start = elapsed_time()
    
    # Run tof through get_kincut_function()
    S1_min, S1_max, S2_max = get_kincut_function(tof)

    # Compare measured energies with maximum/minimum energies for the given time of flight
    accept_inds = np.where((energy_S1 > S1_min) & (energy_S1 < S1_max) & (energy_S2 < S2_max))[0]

    if timer: elapsed_time(t_start, 'kinematic_cuts()')
    return tof[accept_inds], energy_S1[accept_inds], energy_S2[accept_inds]                 


def get_dictionaries(S = 0):
    '''
    Returns empty dictionaries for S1's or S2's which can be used to store data in.\n
    Set S = 'S1' for S1 dictionary
    Set S = 'S2' for S2 dictionary
    Set S = 'merged' for single dictionary with S1 and S2.
    Set S = 'nested' for S2 dictionary nested in S1 dictionary
    S = 0 returns both \n
    
    Example 1: S1_dict = get_dictionaries('S1') returns S1_dict = {'S1_01': [], 'S1_02': [], 'S1_03': [], 'S1_04': [], 'S1_05': []} \n
    Example 2: S1_dict, S2_dict = get_dictionaries() returns both S1_dict = {'S1_01': [], ..., 'S1_05': []} and S2_dict = {'S2_01': [], ..., 'S2_32': []} \n
    Example 3: nested = get_dictionaries('nested') returns nested = {'S1_01': {'S2_01': [], ..., 'S2_32': []}, ..., 'S1_05': {'S2_01': [], ..., 'S2_32': []}} \n
    '''
    
    S1_dictionary = {}
    for i in range(1, 6):
        dict_key = 'S1_0' + str(i)
        S1_dictionary.update({dict_key: []})
    if S == 'S1': return S1_dictionary
    
    S2_dictionary = {}
    for i in range(1, 33):
        if i < 10: dict_key = 'S2_0' + str(i)
        else: dict_key = 'S2_' + str(i)
        S2_dictionary.update({dict_key: []})
    
    if S == 'S2': return S2_dictionary
    if S == 'merged': 
        S1_dictionary.update(S2_dictionary)
        return S1_dictionary
    if S == 'nested': return {'S1_01':get_dictionaries('S2'), 'S1_02':get_dictionaries('S2'), 
                              'S1_03':get_dictionaries('S2'), 'S1_04':get_dictionaries('S2'),
                              'S1_05':get_dictionaries('S2')}
    
    return S1_dictionary, S2_dictionary

def get_boards():
    '''
    Returns an array of board names.
    '''
    return np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])

def get_channels():
    '''
    Returns an array of channel names.
    '''
    return np.array(['A', 'B', 'C', 'D'])

def find_ohmic_phase(shot_number):
    '''
    Returns the time at which the Ohmic phase is over for given shot number.
    '''
    
    # Import NBI info
    nbi = ppf.ppfget(shot_number, dda = "NBI", dtyp = "PTOT")
    nbi_pow = nbi[2]
    nbi_tim = nbi[4]
    
    # Import ICRH info
    icrh = ppf.ppfget(shot_number, dda = "ICRH", dtyp = "PTOT")
    icrh_pow = icrh[2]
    icrh_tim = icrh[4]
    
    # Import LHCD info
    lhcd = ppf.ppfget(shot_number, dda = "LHCD", dtyp = "PTOT")
    lhcd_pow = lhcd[2]
    lhcd_tim = lhcd[4]
    
    
    # No NBI or ICRH or LHCD
    if (len(nbi_pow[nbi_pow > 0])   == 0 and 
        len(icrh_pow[icrh_pow > 0]) == 0 and 
        len(lhcd_pow[lhcd_pow > 0]) == 0): return 70.
    
    # Find where ICRH starts
    if len(icrh_pow[icrh_pow > 0]) > 0: icrh_start = icrh_tim[np.where(icrh_pow != 0)[0][0]]
    else: icrh_start = np.inf
    
    # Find where NBI starts
    if len(nbi_pow[nbi_pow > 0]) > 0: nbi_start = nbi_tim[np.where(nbi_pow != 0)[0][0]]
    else: nbi_start = np.inf
    
    # Find where LHCD starts
    if len(lhcd_pow[lhcd_pow > 0]) > 0: lhcd_start = lhcd_tim[np.where(lhcd_pow !=0)[0][0]]
    else: lhcd_start = np.inf


    first = np.argsort(np.array([nbi_start, icrh_start, lhcd_start]))[0]
    if first == 0: return nbi_start
    elif first == 1: return icrh_start
    elif first == 2: return lhcd_start
    



def elapsed_time(time_start = 0., timed_function = '', return_time = False):
    '''
    Optional timer for functions
    To start timer: 
        t_start = elapsed_time()
    To stop timer and print result:
        elapsed_time(t_start, 'Function name')
    '''
    if not time_start: return time.time()
    else: print('Elapsed time for ' + timed_function + ': ' + '%.2f' %(time.time() - time_start) + ' sec.' )    
    if return_time: return time.time() - time_start



##################################
### Unfinished below this line ###
##################################

#def light_yield_function():
#    '''
#    Plots the light yield function used by Maria Gatu Johnson    
#    '''
#    # Three ranges
#    # Range 1
#    Ep_1 = np.arange(0, 1.91, 0.01)
#    a_1 = 0.0469
#    b_1 = 0.1378
#    c_1 = -0.0183
#    Eee_1 = a_1*Ep_1 + b_1*Ep_1**2 + c_1*Ep_1**3 
#    
#    # Range 2
#    Ep_2 = np.arange(1.9, 9.31, 0.01)
#    a_2 = -0.0142
#    b_2 = 0.1292
#    c_2 = 0.0697
#    d_2 = -0.00315
#    Eee_2 = a_2 + b_2*Ep_2 + c_2*Ep_2**2 + d_2*Ep_2**3
#    
#    # Range 3
#    Ep_3 = np.arange(9.3, 18.91, 0.01)
#    a_3 = -1.8899
#    b_3 = 0.7067
#    Eee_3 = a_3 + b_3*Ep_3
#    
#    plt.figure('Light yield function')
#    plt.plot(Ep_1, Eee_1, 'k')
#    plt.plot(Ep_2, Eee_2, 'k')
#    plt.plot(Ep_3, Eee_3, 'k')
#    plt.xlabel('Proton recoil energy [MeV]')
#    plt.ylabel('Light yield [MeVee]')
#    
    
    
    
def cTOF(S1,S2,dtlow,dthigh,nbins):
    # S1:       time stamps from S1
    # S2:       time stamps from S2
    # dtlow:    time to go back to look for coincidences
    # dthihg:   time to go forward to look for coincidences
    # nbins:    the number of bins within the defined gap
    
    upper = len(S1) - 1
        
    delta = (dthigh - dtlow) / nbins    # time width / bin
    hist = np.zeros(nbins, 'i')
    
        # Define the gap to look between for each event where low is the lower edge and high is the upper edge 
    low = S2 + dtlow
    high = S2 + dthigh  
    
    for i, T in enumerate(S2):
            # Look for the number closest to low[i] in S1 (always rounded up) and store it's index in j
        j = np.searchsorted(S1, low[i])
        
        if j >= upper: 
            break           # Oops, run out of events, kill it all
            
            # If the value in S1 closest to low[i] (the lower edge) is larger than high[i] (the upper edge) then there are no events in the gap
        if S1[j] >= high[i]: 
            continue        # No events in the gap
        
            # In which bin should the event be placed
        L = np.int((S1[j] - low[i]) / delta)
        hist[L] += 1
        
            # Do the same thing another 4 times
        if S1[j + 1] >= high[i]: 
            continue
        
        L = np.int((S1[j + 1] - low[i]) / delta)
        hist[L] += 1         
        
        if S1[j + 2] >= high[i]: 
            continue
        
        L = np.int((S1[j + 2] - low[i]) / delta)
        hist[L] += 1     
        if S1[j + 3] >= high[i]: 
            continue
        
        L = np.int((S1[j + 3] - low[i]) / delta)
        hist[L] += 1       
        
        if S1[j+4] >= high[i]: 
            continue
        
        L = np.int((S1[j + 4] - low[i]) / delta)
        hist[L] += 1    
        
    return hist, delta          


# Plot 1D histogram, allows looping several plots into same window with legend
def hist_1D_s(x_data, title = '', label = 'data set', log = True, bins = 0, ax = -1, 
              normed = 0, density = False, x_label = 'Time [ns]', y_label = '', hist_type = 'step', 
              alpha = 1, linewidth = 1, color = 'k', weights = None, linestyle = '-', timer = False):
    '''
    Example of how to use legend:
    fig = plt.figure('Time differences')
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 1.15 * np.max(dt_ADQ14), 1000)
    hist_1D_s(dt_ADQ412, label = 'ADQ412', log = True, bins = bins, x_label = 'Time difference [ms]', ax = ax)
    hist_1D_s(dt_ADQ14,  label = 'ADQ14',  log = True, bins = bins, x_label = 'Time difference [ms]', ax = ax)
    '''
    if timer: t_start = elapsed_time()
    
    # Create bins if not given
    if bins is 0: bins = np.linspace(np.min(x_data), np.max(x_data), 100)
    

    
    bin_centres = bins[1:] - np.diff(bins) / 2
    hist = np.histogram(x_data, bins = bins, weights = weights)
    if normed: bin_vals = hist[0] / np.max(hist[0])
    else: bin_vals = hist[0]
    
    # Plot crosses instead of step
    if hist_type == 'crosses':
        plt.plot(bin_centres, bin_vals, '+', label = label, color = color, alpha = alpha)
        plt.legend()
        ax = -1
    else:
        plt.hist(bin_centres, bins = bins, weights = bin_vals, log = log,
                 histtype = hist_type, alpha = alpha, linewidth = linewidth,
                 color = color, linestyle = linestyle, density = density)        
#        hist = plt.hist(x_data, label = label, bins = bins, log = log, 
#                    histtype = hist_type, alpha = alpha, 
#                    linewidth = linewidth, weights = weights, color = color, linestyle = linestyle)
    plt.title(title)
    plt.xlim([bins[0], bins[-1]])
    
    plt.xlabel(x_label, fontsize = 14)
    plt.ylabel(y_label, fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    
    # Include legend
    if ax != -1:
        
        handles, labels = ax.get_legend_handles_labels()
#        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
        new_handles = [Line2D([], [], c = color) for h in handles]

        plt.legend(handles=new_handles, labels=labels, loc = 'upper right')
    
    if timer: elapsed_time(t_start, 'hist_1D_s()')
    return hist
#    if return_hist: return hist

def background_subtraction_(disable_cuts, TOF_hist, timer = False):
    '''
    Perform background subtraction of TOF spectrum. If disable_cuts is true an average
    is calculated between -100ns to -50 ns. If disable_cuts is false a model is fit to
    the background and mirrored to the positive TOF side.
    disable_cuts: boolean from user input for disabling/enabling kinematic cuts
    TOF_hist: tuple of histogram information (events, bins)
    '''
    if timer: t_start = elapsed_time()
    
    # Get events and bins
    events = TOF_hist[0]
    bins = TOF_hist[1]
    
    # Without kinematic cuts use average background between -100 ns and -50 ns
    if disable_cuts: 
        tof_bg = np.zeros(len(events))
        tof_bg += np.mean(events[np.where((bins < -50) & (bins > -100))[0]])
    
    # Otherwise fit model to background
    else: 
        def fit_function_1(parameters, bins, data):
            '''
            Fit function for fitting a Gaussian
            '''
            a = parameters[0]
            b = parameters[1]
            c = parameters[2]
                
            # Make gaussian
            A = a * np.exp(-((bins - b) / c)**2)
            diff = A - data
            return diff
        
        def fit_function_2(parameters, bins, data):
            '''
            Fit function for fitting poly2 function. The variable constraint
            ensures that there is a smooth transition from fit_function_1 to
            fit_function_2.
            '''
            a = parameters[0]
            b = parameters[1]
            c = parameters[2]
            end_fit = parameters[3]
            print(end_fit)
            # Find bin/data index corresponding to end_fit
            end_fit = np.argmin(np.abs(bins - end_fit))
            bins = bins[:end_fit]
            data = data[:end_fit]
            # Make poly2
            fcn = a * bins**2 + b * bins + c
            
            diff = fcn - data
            return diff
        
        bin_centres = bins[0:-1] + np.diff(bins)[0]/2
        bin_centres[np.argmin(np.abs(bin_centres))] = 0 # Force central bin to be exactly zero
        
        # Region 0 - negative side of TOF spectrum
        bins_0   = bin_centres[bin_centres < 0]
        events_0 = events[bin_centres < 0]
        
        # Region 1 - Gaussian region
        bin_shift = 8
        bins_1    = bins_0[0:np.argmax(events_0) - bin_shift]
        events_1  = events[0:np.argmax(events_0) - bin_shift]
        
        # Region 2 - Poly2 region
        arg_start_2 = len(bins_1)
        bins_2 = bins_0[arg_start_2:]
        events_2 = events_0[arg_start_2:]
        
        # Region 3 - Zero region
#        bins_3   = bins_0[np.argmin(np.abs(bins_0 + 12)):np.argmin(np.abs(bins_0)) + 1] # bins from -12 ns to 0 ns
        

        
        '''
        Region 1 - Gaussian fit
        '''
        # Starting guesses
        a_1 = 681.3
        b_1 = 18.8
        c_1 = 72.7

        # Fit parameters
        fit_params_1 = optimize.least_squares(fun = fit_function_1, x0 = [a_1, b_1, c_1], args = (bins_1, events_1))['x']
        a_1 = fit_params_1[0]
        b_1 = fit_params_1[1]
        c_1 = fit_params_1[2]
        fit_1 = a_1 * np.exp(-((bins_1 - b_1) / c_1)**2)
        
        '''
        Region 2 - Poly2 fit
        '''
        # Starting guesses
        a_2 = -1.4
        b_2 = -81.4
        c_2 = -770.5
        end_fit = -15.2
        # Fit parameters
        bounds = ((-np.inf, -np.inf, -np.inf, -25), (np.inf, np.inf, np.inf, -10))
        fit_params_2 = optimize.least_squares(fun = fit_function_2, 
                                              x0 = [a_2, b_2, c_2, end_fit], 
                                              args = (bins_2, events_2), 
                                              bounds = bounds)['x']
        a_2 = fit_params_2[0]
        b_2 = fit_params_2[1]
        c_2 = fit_params_2[2]
        end_fit = fit_params_2[3]
        # Find bin corresponding to end_fit
        end_fit = np.argmin(np.abs(bins_0 - end_fit))
        bins_2 = bins_0[arg_start_2:end_fit]
        fit_2 = a_2 * bins_2**2 + b_2 * bins_2 + c_2
        
        '''
        Region 3 - Rolling average
        '''
        # Region 3 - Rolling average region
        bins_3   = bins_0[end_fit:] 
        events_3 = events_0[end_fit:]
        n = 3 # Number of bins to average
        fit_3 = np.array([])
        for n_i in range(len(events_3)):
            if n_i == 0: fit_3 = np.append(fit_3, events_3[0])
            elif n_i == len(events_3) - 1: fit_3 = np.append(fit_3, events_3[-1])
            else: fit_3 = np.append(fit_3, np.sum(events_3[n_i-1:n_i+2] / n))
        
        '''
        Sum of all regions
        '''
        bins_all = np.append(bins_1, bins_2)
        bins_all = np.append(bins_all, bins_3)
        fit_all = np.append(fit_1, fit_2)
        fit_all = np.append(fit_all, fit_3)
        
                
        # Mirror to positive TOF side
        flipped = np.flip(fit_all)
        tof_bg = np.append(fit_all, 0) # Add the zero corresponding to the bin centered at zero
        tof_bg = np.append(tof_bg, flipped)
        
    if timer: elapsed_time(t_start, 'background_subtraction()')
    return tof_bg

def background_subtraction(disable_cuts, TOF_hist, timer = False):
    '''
    Perform background subtraction of TOF spectrum. If disable_cuts is true an average
    is calculated between -100ns to -50 ns. If disable_cuts is false a model is fit to
    the background and mirrored to the positive TOF side.
    disable_cuts: boolean from user input for disabling/enabling kinematic cuts
    TOF_hist: tuple of histogram information (events, bins)
    '''
    if timer: t_start = elapsed_time()
    
    # Get events and bins
    events = TOF_hist[0]
    bins = TOF_hist[1]
    
    # Without kinematic cuts use average background between -100 ns and -50 ns
    if disable_cuts: 
        tof_bg = np.zeros(len(events))
        tof_bg += np.mean(events[np.where((bins < -50) & (bins > -100))[0]])
    
    # Otherwise fit model to background
    else: 
        def fit_function_1(parameters, bins, data):
            '''
            Fit function for fitting a Gaussian
            '''
            a = parameters[0]
            b = parameters[1]
            c = parameters[2]
                
            # Make gaussian
            A = a * np.exp(-((bins - b) / c)**2)
            diff = A - data
            return diff
        
        def fit_function_2(parameters, bins, data):
            '''
            Fit function for fitting poly2 function. The variable constraint
            ensures that there is a smooth transition from fit_function_1 to
            fit_function_2.
            '''
            a = parameters[0]
            b = parameters[1]
            c = parameters[2]
            
            # Make poly2
            fcn = a * bins**2 + b * bins + c
            
            diff = fcn - data
            return diff
        
        bin_centres = bins[0:-1] + np.diff(bins)[0]/2
        bin_centres[np.argmin(np.abs(bin_centres))] = 0 # Force central bin to be exactly zero
        
        # Region 0 - negative side of TOF spectrum
        bins_0   = bin_centres[bin_centres < 0]
        events_0 = events[bin_centres < 0]
        
        # Region 1 - Gaussian region
        bin_shift = 0
        bins_1    = bins_0[0:np.argmax(events_0) - bin_shift]
        events_1  = events[0:np.argmax(events_0) - bin_shift]
        
        # Region 2 - Poly2 region
        arg_start_2 = len(bins_1)
        arg_20 = np.where(np.round(bins_0, 1) == -22.4)[0][0]
        arg_end_2 = np.where(events_0[arg_20:] < 5)[0][0] + arg_20
        bins_2 = bins_0[arg_start_2:arg_end_2] 
        events_2 = events[arg_start_2:arg_end_2]
        
        # Region 3 - Rolling average region
        bins_3   = bins_0[arg_end_2:] 
        events_3 = events_0[arg_end_2:]
        
        '''
        Region 1 - Gaussian fit
        '''
        # Starting guesses
        a_1 = 681.3
        b_1 = 18.8
        c_1 = 72.7

        # Fit parameters
        fit_params_1 = optimize.least_squares(fun = fit_function_1, x0 = [a_1, b_1, c_1], args = (bins_1, events_1))['x']
        a_1 = fit_params_1[0]
        b_1 = fit_params_1[1]
        c_1 = fit_params_1[2]
        fit_1 = a_1 * np.exp(-((bins_1 - b_1) / c_1)**2)
        
        '''
        Region 2 - Poly2 fit
        '''
        # Starting guesses
        a_2 = -1.4
        b_2 = -81.4
        c_2 = -770.5
        # Fit parameters
        fit_params_2 = optimize.least_squares(fun = fit_function_2, x0 = [a_2, b_2, c_2], args = (bins_2, events_2))['x']
        a_2 = fit_params_2[0]
        b_2 = fit_params_2[1]
        c_2 = fit_params_2[2]
        fit_2 = a_2 * bins_2**2 + b_2 * bins_2 + c_2
        
        # Check if rolling average region should be longer (negative values in region 2)
        if not (fit_2 > 0).all():
            # Remove negative values from fit 2
            neg_val = np.where(fit_2 < 0)[0]
            bins_to_transfer = bins_2[neg_val]
            bins_2 = np.delete(bins_2, neg_val)
            fit_2 = np.delete(fit_2, neg_val)
            
            # Add region to rolling average region
            bins_3 = np.insert(bins_3, 0, bins_to_transfer)
            events_3 = np.insert(events_3, 0, events_2[neg_val])
            
            
        '''
        Region 3 - Rolling average
        '''
        n = 4 # Number of bins to average
        fit_3 = np.array([])
        for n_i in range(len(events_3)):
            if n_i == 0: fit_3 = np.append(fit_3, events_3[0])
            elif n_i == len(events_3) - 1: fit_3 = np.append(fit_3, events_3[-1])
            else: fit_3 = np.append(fit_3, np.sum(events_3[n_i-1:n_i+2] / n))
        
        '''
        Sum of all regions
        '''
        bins_all = np.append(bins_1, bins_2)
        bins_all = np.append(bins_all, bins_3)
        fit_all = np.append(fit_1, fit_2)
        fit_all = np.append(fit_all, fit_3)
        
                
        # Mirror to positive TOF side
        flipped = np.flip(fit_all)
        tof_bg = np.append(fit_all, 0) # Add the zero corresponding to the bin centered at zero
        tof_bg = np.append(tof_bg, flipped)
        
    if timer: elapsed_time(t_start, 'background_subtraction()')
    return tof_bg

def plot_2D(times_of_flight, energy_S1, energy_S2, bins_tof = np.arange(-199.8, 200, 0.4), 
            bins_energy = np.arange(-1, 4, 0.02), bins_2D = [np.arange(-199.8, 200, 0.4), np.arange(-1, 4, 0.02)], 
            energy_lim = np.array([-0.1, 2]), tof_lim = np.array([-150, 200]), title = '', 
            log = True, interactive_plot = False, projection = 0, disable_cuts = False,
            times_of_flight_cut = 0, energy_S1_cut = 0, energy_S2_cut = 0, disable_bgs = False, 
            weights = False, hist2D_S1 = None, hist2D_S2 = None, sum_shots = False, proton_recoil = False, timer = False):
    '''
    Plots 2D histogram of TOF vs energy with projections onto time and energy axis.
    Returns nothing.
    times_of_flight: 1D array of times of flight.
    energy_S1: 1D array of energies for S1
    energy_S2: 1D array of energies for S2
    bins_tof: bins for 1D time of flight spectrum
    bins_energy: bins for 1D energy spectrum
    bins_2D: bins for 2D spectrum of energy vs. time of flight
    energy_lim: set energy plotting limits
    title: title
    log: set log scale
    interactive_plot: set to true to set cuts in each spectrum
    projection: used in replot_projections() function, allows for red lines to be plotted along the limits of the cuts
    '''
    if timer: t_start = elapsed_time()
    
    # Add lines for cuts
    if projection != 0: add_lines = True
    else: add_lines = False
    
    fig = plt.figure(title)
    
    '''
    TOF projection
    '''
    # If kinematic cuts are applied
    if not disable_cuts: 
        tof = times_of_flight_cut
        erg_S1 = energy_S1_cut
        erg_S2 = energy_S2_cut
    else: 
        tof = times_of_flight
        erg_S1 = energy_S1
        erg_S2 = energy_S2
    
    # If light yield function is enabled, plot proton recoil energy instead of light yield
    if proton_recoil:
        erg_S1 = inverted_light_yield(erg_S1)
        erg_S2 = inverted_light_yield(erg_S2)
        erg_unit = 'MeV'
    else: erg_unit = 'MeVee'
        
    TOF_fig = plt.subplot(326)
    bins_tof_centres = bins_tof[1:] - np.diff(bins_tof)[0] / 2
    if weights: TOF_hist = plt.hist(bins_tof_centres, bins_tof, weights = tof, log = log, histtype = 'step')
    else: TOF_hist = plt.hist(tof, bins = bins_tof, log = log, histtype = 'step')
    
    # Apply background subtraction
    if not disable_bgs:
        tof_bg_vals = background_subtraction(disable_cuts, TOF_hist)
        
        # Remove background from binned values
        tof_bgs = TOF_hist[0] - tof_bg_vals
        
        # Add to figure
        plt.hist(TOF_hist[1][:-1], bins = TOF_hist[1], weights = tof_bgs, 
                 log = log, histtype = 'step', linestyle = 'dashed', color = 'C1')
        plt.plot(bins_tof_centres, tof_bg_vals, 'r--')
        
    ax_TOF = plt.gca() # Get current axis
    ax_TOF.set_xlabel('Time [ns]')
    ax_TOF.set_ylabel('Counts')
    tof_x_low = tof_lim[0]
    tof_x_high = tof_lim[1]
    ax_TOF.set_xlim([tof_x_low, tof_x_high]) 
    ax_TOF.set_ylim(bottom = np.min(TOF_hist[0]) / 2 + 1)
    
    # Add lines for interactive plot
    if add_lines:
        print('adding lines')
        big_value = 100000000
        if projection['proj'] == 'times-of-flight': 
            proj_lims = projection['limits']
            plt.plot([proj_lims[0], proj_lims[0]], [-big_value, big_value], '--r')
            plt.plot([proj_lims[1], proj_lims[1]], [-big_value, big_value], '--r')
    

    '''
    S1 2D spectrum
    '''
    plt.subplot(322, sharex = ax_TOF)
    
    # Set white background
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)
    
    # Find the max value for the z-axis on the 2D plot (no kinematic cuts applied)
    if weights:
        S1_max = np.max(hist2D_S1)
        S2_max = np.max(hist2D_S2)
    else:
        S1_max = np.max(plt.hist2d(times_of_flight, energy_S1, bins = bins_2D, cmap = my_cmap, vmin = 1)[0])
        S2_max = np.max(plt.hist2d(times_of_flight, energy_S2, bins = bins_2D, cmap = my_cmap, vmin = 1)[0])
    if S1_max >= S2_max: vmax = S1_max
    else: vmax = S2_max
    
    # Plot first 2D histogram (no kinematic cuts applied)
    bins_energy_centres = bins_energy[1:] - np.diff(bins_energy)[0] / 2
    if weights:        
        # Create data set to fill 2D spectrum with one count in each bin
        tof_repeated = np.tile(bins_tof_centres, len(bins_energy_centres))
        energy_repeated = np.repeat(bins_energy_centres, len(bins_tof_centres))
        weights2D_S1 = np.ndarray.flatten(np.transpose(hist2D_S1))
        # Create 2D histogram using weights
        hist2d_S1 = plt.hist2d(tof_repeated, energy_repeated, bins = bins_2D, weights = weights2D_S1,
                               cmap = my_cmap, vmin = 1, vmax = vmax)
    else:
        hist2d_S1 = plt.hist2d(times_of_flight, energy_S1, bins = bins_2D, cmap = my_cmap, vmin = 1, vmax = vmax)
    ax_S1_2D = plt.gca()
    plt.setp(ax_S1_2D.get_xticklabels(), visible = False)
    plt.setp(ax_S1_2D.get_yticklabels(), visible = False)
    
    # Add lines for interactive plot
    if add_lines:
        if projection['proj'] == 'time-of-flight and S1 energy':
            proj_lims_tof = projection['limits'][0]
            proj_lims_E = projection['limits'][1]
            plt.plot([proj_lims_tof[0], proj_lims_tof[0]], [-big_value, big_value], '--r')
            plt.plot([proj_lims_tof[1], proj_lims_tof[1]], [-big_value, big_value], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[0], proj_lims_E[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[1], proj_lims_E[1]], '--r')
        
    # Add lines for kinematic cuts
    if not disable_cuts:
        tof_axis_p = np.linspace(0.1, 500, 500)
        tof_axis_n = np.linspace(-0.1, -500, 500)
        S1_min, S1_max, S2_max = get_kincut_function(tof_axis_p)
        plt.plot(tof_axis_p, S1_min, 'r-')
        plt.plot(tof_axis_p, S1_max, 'r-')
        plt.plot(tof_axis_n, S1_min, 'r-')
        plt.plot(tof_axis_n, S1_max, 'r-')
        
    '''
    S2 2D spectrum
    '''
    # Plot second 2D histogram (no kinematic cuts applied)
    plt.subplot(324, sharex = ax_TOF)
    if weights:        
        # Create 2D histogram using weights
        weights2D_S2 = np.ndarray.flatten(np.transpose(hist2D_S2))
        hist2d_S2 = plt.hist2d(tof_repeated, energy_repeated, bins = bins_2D, weights = weights2D_S2,
                               cmap = my_cmap, vmin = 1, vmax = vmax)   
    else:
        hist2d_S2 = plt.hist2d(times_of_flight, energy_S2, bins = bins_2D, cmap = my_cmap, vmin = 1, vmax = vmax)
    ax_S2_2D = plt.gca()
    plt.setp(ax_S2_2D.get_xticklabels(), visible = False)
    plt.setp(ax_S2_2D.get_yticklabels(), visible = False)
    ax_S2_2D.set_xlim([tof_x_low, tof_x_high])

    if add_lines:
        if projection['proj'] == 'time-of-flight and S2 energy':
            proj_lims_tof = projection['limits'][0]
            proj_lims_E = projection['limits'][1]
            plt.plot([proj_lims_tof[0], proj_lims_tof[0]], [-big_value, big_value], '--r')
            plt.plot([proj_lims_tof[1], proj_lims_tof[1]], [-big_value, big_value], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[0], proj_lims_E[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[1], proj_lims_E[1]], '--r')
    
    # Add lines for kinematic cuts
    if not disable_cuts: 
        plt.plot(tof_axis_p, S2_max, 'r-')
        plt.plot(tof_axis_n, S2_max, 'r-')
    '''
    Colour bar
    '''
    ax_colorbar = fig.add_axes([0.12, 0.2, 0.28, 0.03])
    plt.colorbar(ax = ax_S1_2D, cax = ax_colorbar, orientation = 'horizontal')
    

    
    '''
    S2 energy projection
    '''
    plt.subplot(323, sharey = ax_S2_2D)
    if weights: S2_E_hist = plt.hist(bins_energy_centres, bins = bins_energy, weights = erg_S2,
                                     orientation = 'horizontal', histtype = 'step', log = log)
    else: S2_E_hist = plt.hist(erg_S2, bins = bins_energy, orientation = 'horizontal', 
                               histtype = 'step', log = log)
    ax_S2_E = plt.gca()
    ax_S2_E.set_xlabel('Counts')
    y_lower = energy_lim[0]
    y_upper = energy_lim[1]

    if add_lines:
        if projection['proj'] == 'S2':
            proj_lims = projection['limits']
            plt.plot([-big_value, big_value], [proj_lims[0], proj_lims[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims[1], proj_lims[1]], '--r')
    ax_S2_E.set_ylim([y_lower, y_upper])
#    ax_S2_E.set_xlim(xmin = np.min(S2_E_hist[0][S2_E_hist[0] > 0]))
    '''
    S1 energy projection
    '''
    plt.subplot(321, sharey = ax_S1_2D, sharex = ax_S2_E)
    
    if add_lines:
        if projection['proj'] == 'S1':
            proj_lims = projection['limits']
            plt.plot([-big_value, big_value], [proj_lims[0], proj_lims[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims[1], proj_lims[1]], '--r')
    
    if weights: S1_E_hist = plt.hist(bins_energy_centres, bins = bins_energy, 
                                     weights = erg_S1, orientation = 'horizontal',
                                     histtype = 'step', log = log)
    else: S1_E_hist = plt.hist(erg_S1, bins = bins_energy, orientation = 'horizontal', 
                               histtype = 'step', log = log)
    ax_S1_E = plt.gca()
    plt.setp(ax_S1_E.get_xticklabels(), visible = False)
    ax_S1_E.set_ylim([y_lower, y_upper])
    
    # Set the x-axis limits
    x_lower = 0.1
    S2_events = S2_E_hist[0]
    S1_events = S1_E_hist[0]
    if np.sum(S2_events) == 0: x_upper = 1
    elif np.max(S2_events) >= np.max(S1_events): x_upper = np.max(S2_events)
    else: x_upper = np.max(S1_events)
    
    if np.sum(S2_events) == 0 or np.sum(S1_events) == 0: x_lower  = 0
    elif np.min(S2_events[S2_events > 0] <= np.min(S1_events[S1_events > 0])): 
        x_lower = np.min(S2_events[S2_events > 0])
    else: x_lower = np.min(S1_events[S1_events > 0])
    ax_S2_E.set_xlim([x_lower, x_upper])
    
    # Set x,y-label
    fig.text(0.04, 0.65, f'Deposited energy [{erg_unit}]', va='center', rotation='vertical')
    fig.text(0.5, 0.93, title, va = 'center', ha = 'center')
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    
    '''
    Begin interactive plotting
    '''
    plt.show(block = False)
    if interactive_plot:
        plt.show(block = False)
        while True:
            print('\nSelect one of the following panels and set upper and lower limits to project the selected limit onto the other dimensions.')
            print('TL - top left')
            print('TR - top right')
            print('ML - middle left')
            print('MR - middle right')
            print('BR - bottom right')
            print('Type exit to exit.')
            panel_choice = input('Select panel (TL, TR, ML, MR, BR): ')
            if panel_choice in ['exit', 'Exit', 'EXIT']: break
            
            # Cut in 2D Spectrum
            if panel_choice in ['TR', 'tr', 'MR', 'mr']:
                tof_choice = input('Type limits for time axis ("lower upper"): ')
                energy_choice = input('Type limits for energy axis ("lower upper"): ')
                
                # Find space in user input
                find_space_tof = tof_choice.find(' ')
                find_space_energy = energy_choice.find(' ')
                
                # Transform to array of two floats [float_1, float_2]
                tof_choice = [float(tof_choice[0:find_space_tof]), float(tof_choice[find_space_tof + 1:])]
                energy_choice = [float(energy_choice[0:find_space_energy]), float(energy_choice[find_space_energy + 1:])]
                limits = [tof_choice, energy_choice]    
            
            # Cut in 1D spectrum
            elif panel_choice in ['TL', 'tl', 'ML', 'ml', 'BR', 'br']:
                if panel_choice in ['BR', 'br']: limits = input('Type limits for time axis ("lower upper"): ')
                elif panel_choice in ['TL', 'tl']: limits = input('Type limits for S1 energy axis ("lower upper"): ')
                else: limits = input('Type limits for S2 energy axis ("upper lower"): ')    
                
                # Find space in user input
                find_space  = limits.find(' ')
                limits = [float(limits[0:find_space]), float(limits[find_space + 1:])]
                
            else: 
                print('Invalid choice.')
                continue
            # Replot with new projections
            replot_projections(limits = limits, panel_choice = panel_choice, times_of_flight = tof,
                               energy_S1 = erg_S1, energy_S2 = erg_S2, bins_tof = bins_tof, bins_energy = bins_energy, 
                               bins_2D = bins_2D, log = log, disable_cuts = disable_cuts, disable_bgs = True, 
                               energy_S1_cut = erg_S1, energy_S2_cut = erg_S2, times_of_flight_cut = tof, proton_recoil = False)
    elif sum_shots: plt.close(fig)

                
        


    if timer: elapsed_time(t_start, 'plot_2D()')
    return TOF_hist, S1_E_hist, S2_E_hist, hist2d_S1, hist2d_S2

    
def replot_projections(limits, panel_choice, times_of_flight, energy_S1, energy_S2, bins_tof, bins_energy, bins_2D, 
                       log = True, disable_cuts = False, disable_bgs = False, energy_S1_cut = 0, energy_S2_cut = 0, 
                       times_of_flight_cut = 0, proton_recoil = False):
    '''
    Replot the spectra with a cut on one of the energy projections
    limits: limits of cuts for projections. 1x2 array for 1D spectrum, 2x2 array for 2D spectrum
    panel_choice: panel to be cut (BR - bottom right, ML - middle left, TR - top right etc.)
    times_of_flight: 1D array of times of flight
    energy_S1: 1D array of energies for S1
    energy_S2: 1D array of energies for S2
    '''
    
    # If cut has been made in one of the 1D spectra
    if np.shape(limits) == (2, ):
        # Make cut in S1 or S2 energy
        if panel_choice in ['TL', 'tl']: 
            if not disable_cuts: eoi = energy_S1_cut
            else: eoi = energy_S1
            det = 'S1'
            uni = 'E'
            proj = {'S1_energy':[]}
        if panel_choice in ['ML', 'ml']: 
            if not disable_cuts: eoi = energy_S2_cut
            else: eoi = energy_S2
            det = 'S2'
            uni = 'E'

        if panel_choice in ['BR', 'br']:
            if not disable_cuts: eoi = times_of_flight_cut
            else: eoi = times_of_flight
            det = 'times-of-flight'
            uni = 'tof'
        
        # Find all events within this cut
        inds = np.where((eoi >= limits[0]) & (eoi <= limits[1]))[0]
        
        
#        title = 'Cut in ' + det + '\n' + str(round(limits[0], 2)) + ' < ' + uni + ' < ' + str(round(limits[1], 2))
        title = f'Cut in {det}\n{round(limits[0], 2)} < {uni} < {round(limits[1], 2)}'
        
    # If the cut has been made in one of the 2D spectra
    else:
        if panel_choice in ['TR', 'tr']:
            eoi = energy_S1
            det = 'time-of-flight and S1 energy'
        elif panel_choice in ['MR', 'mr']:
            eoi = energy_S2
            det = 'time-of-flight and S2 energy'
        
        # Select events which fulfill the criteria in tof and energy
        tof_choice = limits[0]
        energy_choice = limits[1]
        
        inds = np.where((times_of_flight >= tof_choice[0]) & (times_of_flight <= tof_choice[1]) &
                        (eoi >= energy_choice[0]) & (eoi <= energy_choice[1]))
        title = f'Cut in {det}\n{round(tof_choice[0], 2)} < tof < {round(tof_choice[1], 2)}, {round(energy_choice[0], 2)} < E < {round(energy_choice[1], 2)}'

       
    proj = {'proj':det, 'limits':limits}
    
    # Replot using the cut
    times_of_flight = times_of_flight[inds]
    energy_S1 = energy_S1[inds]
    energy_S2 = energy_S2[inds]
    
    plot_2D(times_of_flight = times_of_flight, energy_S1 = energy_S1, 
              energy_S2 = energy_S2, bins_tof = bins_tof, bins_energy = bins_energy,
              times_of_flight_cut = times_of_flight, energy_S1_cut = energy_S1, energy_S2_cut = energy_S2_cut,
              bins_2D = bins_2D, interactive_plot = False, disable_cuts = disable_cuts, 
              disable_bgs = disable_bgs, title = title, projection = proj, log = log, proton_recoil = proton_recoil)
    
    
def print_help():
    print('\nPlease supply the shot number.')
    print('Example: python create_TOF.py --JPN 94217')
    print('Set the shot number to 0 to run the latest shot.')
    print('\nAdditional optional arguments:')
    print('--input-file my_input.txt: Input arguments from my_input.txt are used.')
    print('--1D-spectrum: Only plot time-of-flight spectrum.')
    print('--remove-doubles mode:\n  \
mode = 0: Remove all events which have produced a coincidence between two S1\'s.\n  \
mode = 1: Only plot events which have produced a coincidence between two S1\'s')
    print('--save-data my_file_name: Save the data as a python pickle with file name \"my_file_name\".')
    print('--time-range start stop: Only plot the data between \"start\" and \"stop\" seconds into the shot. \"start\" and \"stop\" are given in number of seconds since PRE.')
    print('--disable-cuts: Plot the data without any kinematic cuts.')
    print('--disable-bgs: Plot the data without background subtracting the time-of-flight spectrum.')
    print('--disable-detectors: Analysis is not performed on detectors specified by user. Example: --disable-detectors S1_01,S1_02,S2_01')
    print('--enable-detectors: Analysis is only performed on detectors speecified by user. Example: --enable-detectors S1_01,S2_01')
    print('--ohmic-spectrum: Analysis is only performed for the Ohmic phase of the shot.')
    print('--run-timer: Print the elapsed time for each function.')
    print('--set-thresholds thr_l thr_u: Set lower and upper energy thresholds where \"thr_l\" and \"thr_u\" are given in MeVee. If \"thr_u\" is omitted it is set to +inf.')
    print('--help: Print this help text.')
    
