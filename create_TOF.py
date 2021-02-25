#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 07:41:37 2021

@author: beriksso
"""

'''
Produces a time of flight spectrum from given shot number
'''

import sys
from functions import tofu_functions as dfs
#import definitions as dfs
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import ppf
import pickle
import os

def set_timer_level(detector_name):
    # Set timer level
    if time_level in s1_dicts.keys() or time_level in s2_dicts.keys():
        if time_level == detector_name: timer_level = 1
        else: timer_level = 0
    else: timer_level = time_level

    return timer_level

def get_S1_times(arguments):
    '''
    Used in main function for parallelization of grabbing 
    times for removing coincidences between S1's
    '''
    boa = arguments[0:2]
    cha = arguments[3]
    det_name = dfs.get_detector_name(board = boa, channel = cha)
    if det_name in disabled_detectors: return np.array([])
    if det_name not in enabled_detectors: return np.array([])
    shot_number = int(arguments[5:])
    
    return dfs.get_times(board = boa, channel = cha, shot_number = shot_number, timer = time_level)
    
def get_doubles(arguments):
    '''
    Used in main function for parallelization of finding
    coincidences between S1's
    '''
    a = arguments[0:5]
    b = arguments[6:]
    
    # Set timer level
    s1a_timer_level = set_timer_level(a)
    s1b_timer_level = set_timer_level(b)
    timer_level = s1a_timer_level or s1b_timer_level

    if a in disabled_detectors or b in disabled_detectors: return np.array([])
    if a not in enabled_detectors or b not in enabled_detectors: return np.array([])
    s1_a = S1_times[a]
    s1_b = S1_times[b]
    t_test = dfs.elapsed_time()
    _, inds = dfs.sTOF4(np.array(s1_a), np.array(s1_b), t_back = 10, t_forward = 10, return_indices = True, timer = timer_level)
    dfs.elapsed_time(t_test, 'First call')
    print(f'{a}, {b} done.')
    return inds

def get_tof(arguments):
    '''
    Used in main function for parallelization of finding 
    coincidences between S1's and S2's
    '''
    s1 = arguments[0][0:5]
    s2 = arguments[0][6:]
    
    s1_timer_level = set_timer_level(s1)
    s2_timer_level = set_timer_level(s2)
    timer_level = s1_timer_level or s2_timer_level
    
    if s1 in disabled_detectors or s2 in disabled_detectors: return np.array([]), np.array([]), s1, s2
    if s1 not in enabled_detectors or s2 not in enabled_detectors: return np.array([]), np.array([]), s1, s2
    
    s1_times = arguments[1][s1]
    s2_times = arguments[1][s2]
    t_test = dfs.elapsed_time()
    tof, inds = dfs.sTOF4(s1_times, s2_times, t_back = time_window, t_forward = time_window, return_indices = True, timer = timer_level)
    dfs.elapsed_time(t_test, 'Second call')

    return tof, inds, s1, s2

def import_all_data(arguments):
    '''
    Used in main function for parallelization of importing time stamps
    and pulse data.
    '''
    
    # Get all arguments
    detector_name = arguments[0:5]
    shot_number = int(arguments[5:])

    # Set timer level
    timer_level = set_timer_level(detector_name)
    if timer_level: t_start = dfs.elapsed_time()
    
    # Get board and channel for given detector
    boa, cha = dfs.get_board_name(detector_name, timer = timer_level)

    # Skip detectors requested by user
    if detector_name in disabled_detectors:
        print(f'{detector_name}: Disabled')
        return np.array([]), detector_name, np.array([])

    if detector_name not in enabled_detectors: 
        print(f'{detector_name}: Disabled')
        return np.array([]), detector_name, np.array([])
    
    # Import time stamps
    time_stamps_LED = dfs.get_times(board = boa, channel = cha, shot_number = shot_number, timer = timer_level)
    if np.shape(time_stamps_LED) == (): 
        print(detector_name + ': Missing data')
        return np.array([]), detector_name, np.array([])
    
    # Import time offsets from INI to PRE for ADQ412's
    if boa in ['06', '07', '08', '09', '10']:
        # Import the time offset for ADQ412
        time_offset = dfs.get_offset(boa, shot_number, timer = timer_level)
    else: time_offset = 0
    
    # Apply time offsets to time stamps
    time_stamps_LED -= time_offset
    
    # Find the time range in time_stamps_LED
    pulse_start = np.searchsorted(time_stamps_LED, time_slice[0] * 1E+9)
    pulse_end   = np.searchsorted(time_stamps_LED, time_slice[1] * 1E+9)

    # Select appropriate chunk of time stamps
    time_stamps = time_stamps_LED[pulse_start:pulse_end]
    if detector_name[0:2] == 'S2' and len(time_stamps_LED) / 8. > 70000: 
        print('WARNING: Suspiciously large number of pulses on ' + detector_name + ', LED may have been switched on during pulse.')

    # If there is no data in time_stamps within this range return empty arrays
    if len(time_stamps) == 0: return np.array([]), detector_name, np.array([])
    
    # Import pulse data and remove the chunk we don't need 
    pulse_data = dfs.get_pulses(boa, cha, shot_number, pulse_start = pulse_start, pulse_end = pulse_end, timer = timer_level)    
    
    print(f'{detector_name} data downloaded.')
    if timer_level: dfs.elapsed_time(t_start, 'import_all_data()')
    return time_stamps, detector_name, pulse_data

def create_TOF(arguments):
    '''
    Analysis of pulse waveforms to modify time stamp using time-pickoff method.
    Run in parallel
    '''
    
    if time_level: t_start = dfs.elapsed_time()
    
    # Get all arguments
    boa = arguments[0:2]
    cha = arguments[3]
    shot_number = int(arguments[5:])
    
    # Skip the channels without data
    detector_name = dfs.get_detector_name(boa, cha)
    if detector_name in ['ABS_REF', '1kHz_CLK', 'DEAD']:
        return 0, detector_name

    # Skip detectors requested by user
    if detector_name in disabled_detectors:
        print(f'{detector_name}: Disabled')
        return np.array([]), detector_name, np.array([])

    if detector_name not in enabled_detectors: 
        print(f'{detector_name}: Disabled')
        return np.array([]), detector_name, np.array([])

    # Set timer level
    if time_level in s1_dicts.keys() or time_level in s2_dicts.keys():
        if time_level == detector_name: timer_level = 1
        else: timer_level = 0
    else: timer_level = time_level

    # Select data set corresponding to detector
    time_data = time_stamps[detector_name]
    pulse_data = pulses[detector_name]

    # Find number of pre trigger samples
    pre_trig_samples = dfs.get_pre_trigger(boa, shot_number, timer = timer_level)
    if boa in ['01', '02', '03', '04', '05']:
        # Find adjustment to number of pre trigger samples for ADQ14 pulse data
        '''
        We ask for 16 pre trigger samples but receive anything between 16-19 
        (feature of the fwpd_disk_stream software), but since we know the trigger level
        used we can find the number of pre trigger samples.
        '''
        
        # First get trigger level
        trig_level = dfs.get_trigger_level(boa, cha, shot_number, timer = timer_level)

        # Get pre trigger adjustment
        pre_trig_adjustment = dfs.find_threshold(pulse_data, trig_level = trig_level, timer = timer_level, detector_name = detector_name)            

        # Find where pulses trigger as normal
        normal_trig = np.where(pre_trig_adjustment > pre_trig_samples - 1)[0] 

        # Remove oddly triggering pulses
        time_data  = time_data[normal_trig]
        pulse_data = pulse_data[normal_trig]
        pre_trig_adjustment = pre_trig_adjustment[normal_trig]
   
    elif boa in ['06', '07', '08', '09', '10']:        
        # First get trigger level
        trig_level = dfs.get_trigger_level(boa, cha, shot_number, timer = timer_level)
        pre_trig_adjustment = dfs.find_threshold(pulse_data, trig_level = trig_level, timer = timer_level)            
        # Find where pulses trigger as normal
        normal_trig = np.where(pre_trig_adjustment > pre_trig_samples - 1)[0] 
        
        # Remove oddly triggering pulses
        time_data = time_data[normal_trig]
        pulse_data = pulse_data[normal_trig]
        pre_trig_adjustment = pre_trig_adjustment[normal_trig]

    # Perform baseline reduction
    pulse_data_bl = dfs.baseline_reduction(pulse_data, timer = timer_level)

    # Remove junk pulses and corresponding times
    bias_level = dfs.get_bias_level(shot_number = shot_number, board = boa)
    pulse_data_bl, good_indices = dfs.cleanup(pulses = pulse_data_bl, dx = 1, bias_level = bias_level, detector_name = detector_name)
    pulse_data = pulse_data[good_indices]
    time_data = time_data[good_indices]
    pre_trig_adjustment = pre_trig_adjustment[good_indices]

    # Set up x-axes for sinc interpolation
    u_factor = 10
    record_length = np.shape(pulse_data)[1]
    x_axis = np.arange(0, record_length)
    ux_axis = np.arange(0, record_length, 1./u_factor)
    
    # Perform sinc interpolation
    pulse_data_sinc = dfs.sinc_interpolation(pulse_data_bl, x_axis, ux_axis, timer = timer_level)
    
    # If 2D plotting enabled
    if not plot_1D: 
        
        # Get area under pulse
        pulse_area = dfs.get_pulse_area(pulse_data_sinc, u_factor, timer = timer_level)
        
        # Convert to deposited energy
        pulse_energy = dfs.get_energy_calibration(-pulse_area, detector_name, timer = timer_level)
        
        # Remove pulses outside user defined energy thresholds
        pulse_energy_thr = np.where((pulse_energy > thr_l) & (pulse_energy < thr_u))[0]
        pulse_energy = pulse_energy[pulse_energy_thr]
        pulse_data_sinc = pulse_data_sinc[pulse_energy_thr]
        time_data = time_data[pulse_energy_thr]
        pre_trig_adjustment = pre_trig_adjustment[pulse_energy_thr]
    else: pulse_energy = 0
    # Perform time pickoff method
    time_pickoff = dfs.time_pickoff_CFD(pulse_data_sinc, fraction = 0.05 ,timer = timer_level) * 1. / u_factor

    # Calculate how much to adjust the ADQ14 time stamps by
    if boa in ['01', '02', '03', '04', '05']: time_adjustment = time_pickoff - pre_trig_adjustment
    # Calculate how much to adjust the ADQ412 time stamps
    elif boa in ['06', '07', '08', '09', '10']: time_adjustment = time_pickoff
    # Calculate the new times and append to S1/S2 timelist
    new_times = time_data + time_adjustment
    if timer_level: dfs.elapsed_time(t_start, detector_name)
    print(detector_name + ': Done')
    return new_times, detector_name, pulse_energy




if __name__=="__main__":
    t_start = dfs.elapsed_time()

    # Constants, arguments
    time_level              = 0
    time_slice              = np.array([0., 0.])
    time_range_file         = 0
    plot_1D                 = False
    remove_doubles          = False
    doubles_mode            = -1
    sys_exit                = False
    skip_flag               = 0
    save_data               = False
    save_NES                = False
    warn_LED                = False
    disable_cuts            = False
    disable_bgs             = False
    disable_scratch         = False
    ohmic_spectrum          = False
    interactive_plot        = True
    sum_shots               = False
    proton_recoil           = False
    shots                   = np.array([])
    disabled_detectors      = []
    s1_dicts , s2_dicts     = dfs.get_dictionaries()
    enabled_detectors       = dfs.get_dictionaries('merged')
    bins                    = np.arange(-199.8, 200, 0.4)
    bins_energy             = np.arange(-1, 8, 0.02)
    tof_vals                = np.zeros(len(bins) - 1)
    erg_S1_vals             = np.zeros(len(bins_energy) - 1)
    erg_S2_vals             = np.zeros(len(bins_energy) - 1)
    processed_shots         = np.array([])
    thr_l                   = 0
    thr_u                   = np.inf
    E_low                   = -0.1
    E_high                  = 2
    shift_file              = 'shift_files/shift_V4.txt'
    time_window             = 500 # Time window (+-) for TOF spectrum [ns]

    
    if len(sys.argv) == 1: 
        dfs.print_help()
        sys.exit
    else:        
        i = 1
        # Parse through arguments
        while i < len(sys.argv):
            if skip_flag:
                skip_flag -= 1
                i += 1
                continue
            
            # Check argument validity
            if sys.argv[i][0:2] != '--': 
                error_message = 'Invalid argument. Use --help for further information.'
                sys_exit = True
            
            # Shot number
            elif sys.argv[i] == '--JPN':
                if i == len(sys.argv) - 1:
                    error_message = '--JPN requires an additional argument.'
                    sys_exit = True
                else: 
                    shots = np.array([sys.argv[i + 1]])
                    try: 
                        int(shots[0])
                        if shots[0] == '0':
                            # Get latest shot
                            shots = np.array([str(ppf.pdmsht())])
                    except: 
                        error_message = 'JPN provided incorrectly.'
                        sys_exit = True
                    skip_flag = 1
            
            # Read input file
            elif sys.argv[i] == '--input-file':
                if sys.argv[i + 1][0:2] == '--': 
                    error_message = '--input-file requires an additional argument.'
                    sys_exit = True
                else:
                    # Read input file
                    with open(sys.argv[i + 1]) as handle:
                        c = handle.readlines()
                    for arg in c:
                        if arg[0] == '#': continue
                        elif ' ' not in arg: sys.argv = np.append(sys.argv, arg[:-1])
                        else:
                            save_num = 0
                            for num, t in enumerate(arg):
                                if t == ' ':
                                    if save_num == 0: sys.argv = np.append(sys.argv, arg[0:num])
                                    else: sys.argv = np.append(sys.argv, arg[save_num+1:num])
                                    save_num = num
                                elif t == '\n':
                                    sys.argv = np.append(sys.argv, arg[save_num+1:-1])
                                    break
                    skip_flag = 1
            
            # Plot only 1D spectrum
            elif sys.argv[i] == '--1D-spectrum': 
                plot_1D = True
                disable_cuts = True
            
            # Timer for every function
            elif sys.argv[i] == '--run-timer': 
                try: 
                    tl = sys.argv[i+1][0:2]
                    if tl == '--': time_level = 1
                    elif sys.argv[i+1] in s1_dicts.keys() or sys.argv[i+1] in s2_dicts.keys():
                        time_level = sys.argv[i+1]
                        skip_flag = 1
                    else: 
                        error_message = 'Invalid argument for --run-timer.'
                        sys_exit = True
                except: time_level = 1
            
            # Save all data to file   
            elif sys.argv[i] == '--save-data':
                file_name = sys.argv[i + 1]
                save_data = True
                interactive_plot = False
                skip_flag = 1
            
            # Save histogram data to file
            elif sys.argv[i] == '--save-NES': 
                if i == len(sys.argv) - 1:
                    error_message = '--save-NES requires an additional argument.'
                    sys_exit = True
                elif sys.argv[i + 1][0:2] == '--': 
                    error_message = '--save-NES requires an additional argument.'
                    sys_exit = True
                else:
                    save_NES = True
                    filename_NES = sys.argv[i + 1]
                    interactive_plot = False
                    skip_flag = 1
            
            # Remove/keep double scattering events in S1
            elif sys.argv[i] == '--remove-doubles': 
                try: 
                    doubles_mode = int(sys.argv[i + 1])
                    if doubles_mode not in [0, 1]: 
                        error_message = 'Invalid mode for --remove-doubles.'
                        sys_exit = True
                except: 
                    error_message = '--remove-doubles requires an additional argument.'
                    sys_exit = True
                remove_doubles = True
                skip_flag = 1
            
            # Set time range of interest
            elif sys.argv[i] == '--time-range':
                time_slice[0] = np.double(sys.argv[i + 1])
                time_slice[1] = np.double(sys.argv[i + 2])
                skip_flag = 2
            
            # Read time ranges from file
            elif sys.argv[i] == '--time-range-file':
                content = np.loadtxt(sys.argv[i + 1], ndmin = 2)
                shots = np.array(content[:, 0], dtype = 'int')
                shots = np.array(shots, 'str')
                t1    = content[:, 1]
                t2    = content[:, 2]
                # Create dictionary of shots and time ranges
                time_range_file = {}
                for enum, sn in enumerate(shots):
                    if sn not in time_range_file.keys(): time_range_file[sn] = [[t1[enum], t2[enum]]]
                    else: time_range_file[sn].append([t1[enum], t2[enum]])
                skip_flag = 1
            
            # Disable kinematic cuts
            elif sys.argv[i] == '--disable-cuts': disable_cuts = True
            
            # Disable background subtraction
            elif sys.argv[i] == '--disable-bgs': disable_bgs = True
            
            # Disable get data from scratch
            elif sys.argv[i] == '--disable-scratch': disable_scratch = True
            
            # Disable given detectors
            elif sys.argv[i] == '--disable-detectors': 
                if sys.argv[i + 1][0:2] == '--': 
                    error_message = '--disable-detectors requires an additional argument.'
                    sys_exit = True
                else: 
                    disabled_detectors = sys.argv[i + 1]
                    skip_flag = 1
            
            # Enable given detectors
            elif sys.argv[i] == '--enable-detectors': 
                if sys.argv[i + 1][0:2] == '--':
                    error_message = '--enable-detectors requires an additional argument.'
                    sys_exit = True
                else:
                    enabled_detectors = sys.argv[i + 1]
                    skip_flag = 1
            
            # Only plot Ohmic phase
            elif sys.argv[i] == '--ohmic-spectrum': ohmic_spectrum = True
            
            # Enable light yield function
            elif sys.argv[i] == '--proton-recoil-energy': 
                proton_recoil = True
                bins_energy = np.arange(0, 10, 0.01)
                erg_S1_vals = np.zeros(len(bins_energy) - 1)
                erg_S2_vals = np.zeros(len(bins_energy) - 1)
            
            # Sum several shots
            elif sys.argv[i] == '--sum-shots':
                interactive_plot = False
                sum_shots = True
                j = 0
                while i + j + 1 < len(sys.argv):
                    if sys.argv[i + j + 1][0:2] == '--': break
                    shots = np.append(shots, sys.argv[i + j + 1])
                    j += 1
                skip_flag = j
            
            elif sys.argv[i] == '--set-thresholds':
                if sys.argv[i + 1][0:2] == '--':
                    error_message = '--set-thresholds requires at least one additional argument.'
                    sys_exit = True
                try: 
                    next_arg = sys.argv[i + 2][0:2] == '--'
                    if next_arg:
                        thr_l = np.float(sys.argv[i + 1])
                        skip_flag = 1
                    else:
                        thr_l = np.float(sys.argv[i + 1])
                        thr_u = np.float(sys.argv[i + 2])
                        E_high = thr_u
                        skip_flag = 2
                except: 
                    thr_l = np.float(sys.argv[i + 1])
                    skip_flag = 1

                print(f'Thresholds set to {thr_l} < E [MeVee] < {thr_u}')
            
            # Print help message
            elif sys.argv[i] == '--help': 
                dfs.print_help()
                sys.exit()
            else: 
                error_message = 'Invalid argument: ' + sys.argv[i] +  '.\nUse --help for further information.'
                sys_exit = True
            i += 1

    # Give information to user
    if sys_exit: raise Exception(error_message)
        
    
    # Check number of availble cores
    available_cpu = mp.cpu_count()

    for counter, shot_number in enumerate(shots):
        print('Shot number:', shot_number)
        
        # Check if data available in scratch
        scratch_path = f'/common/scratch/beriksso/TOFu/data/{shot_number}/'
        scratch_exists = os.path.exists(scratch_path)


        # If scratch disabled perform full analysis
        if disable_scratch: full_analysis = True
        # Otherwise if scratch exists let user choose
        elif scratch_exists: 
            ans = input('Use data available on scratch (some functions may not work as expected)? [y/n] ')
            if ans in ['n', 'N']: full_analysis = True
            if ans in ['y', 'Y']: full_analysis = False
        # If scratch does not exist perform full analysis
        else: full_analysis = True

        if full_analysis:
            if available_cpu > 16: available_cpu = 16
            print('Running on ' + str(available_cpu) + ' cores.')
            '''
            Find which part of the shot we want to work with
            '''    
            if sum_shots: time_slice = [0, 0]
            # If no time range is given by user
            if time_slice[0] == 0 and time_slice[1] == 0 and time_range_file == 0:
                time_slice = dfs.find_time_range(shot_number)
            # If range is given from file
            elif time_range_file != 0:
                time_slice = time_range_file[shot_number][0]
                time_range_file[shot_number].pop(0)
            
            # If Ohmic phase is requested
            if ohmic_spectrum: 
                t_ohmic = dfs.find_ohmic_phase(shot_number)
                # Set time slice end to 0.1 s before NBI/RF kicks in.
                time_slice = np.array([40., t_ohmic - 0.1])
                
            '''
            Perform coincidence analysis on raw time stamps, anything that
            does not produce coincidence is removed.
            '''            
            # Arguments for parallelization of importing all data 
            data_argu = [sx + f' {shot_number}' for sx in dfs.get_dictionaries('merged').keys()]
            
            print('Collecting data from LPF...')
            with mp.Pool(available_cpu) as pool_data:
                data_info = pool_data.map(import_all_data, data_argu)
            
            # Dictionary to store time stamps and pulses in
            time_stamps = {di[1]:di[0] for di in data_info}
            pulses      = {di[1]:di[2] for di in data_info}
            
            # Arguments for parallelization of finding coincidences between S1's and S2's                
            tof_argu = [[f'{s1} {s2}', time_stamps] for s1 in s1_dicts.keys() for s2 in s2_dicts.keys()]
            
            # Dictionary to store coincidences in
            coincidences_raw = dfs.get_dictionaries('nested')
            indices_raw = dfs.get_dictionaries('nested')
            print('Removing non-coincident pulses.')
            with mp.Pool(available_cpu) as pool_tof:
                tof_info = pool_tof.map(get_tof, tof_argu)
                
            # Only keep time stamps and pulses which produce coincidences
            long_indices = dfs.get_dictionaries('merged')
            for ti in tof_info:
                long_indices[ti[2]] = np.append(long_indices[ti[2]], ti[1][0])
                long_indices[ti[3]] = np.append(long_indices[ti[3]], ti[1][1])
            indices = {key:np.array(np.unique(value), dtype = 'int') for key, value in long_indices.items()}
            for sx, ind in indices.items():
                time_stamps[sx] = time_stamps[sx][ind]
                pulses[sx] = pulses[sx][ind]
            
            # Get shifts for aligning S1's vs S2's
            shifts = dfs.get_shifts(shift_file, time_level)
            
            # Remove all coincidence events between two S1's
            if remove_doubles:
                S1_indices = {'S1_01':np.array([], dtype = 'int64'), 
                              'S1_02':np.array([], dtype = 'int64'),
                              'S1_03':np.array([], dtype = 'int64'),
                              'S1_04':np.array([], dtype = 'int64'),
                              'S1_05':np.array([], dtype = 'int64')}
                
                # Get shifts for aligning S1's vs S1-5
                A = np.loadtxt(shift_file, dtype = 'str')
                S1_shifts = np.array(A[-4:, 1], dtype = 'float')
                
                S1_times = dfs.get_dictionaries('S1')
                # Store in dictionary and shift
                for i, (sx, times) in enumerate(time_stamps.items()):
                    if sx not in ['S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05']:
                        continue
                    elif sx == 'S1_05': S1_shift = 0
                    else:  S1_shift = S1_shifts[i]
                    if np.shape(times) == (): 
                        S1_times[sx] = np.array([])
                        continue
                    S1_times[sx] = time_stamps[sx] + S1_shift
                    
                    
                # Find coincidences
                print('\nFinding coincidences between S1\'s')        
                tof_argu = ['S1_01 S1_02', 'S1_01 S1_03', 'S1_01 S1_04', 'S1_01 S1_05',
                            'S1_02 S1_03', 'S1_02 S1_04', 'S1_02 S1_05',
                            'S1_03 S1_04', 'S1_03 S1_05',
                            'S1_04 S1_05']
#                tof_argu = [[s1_combo, S1_times] for s1_combo in tof_argu]
                
                with mp.Pool(available_cpu) as pool_t:
                    coincidence_indices = pool_t.map(get_doubles, tof_argu) 
                
                # Pick out unique indices and order in dictionary
                for i, ta in enumerate(tof_argu):
                    # Choose detector name from tof_argu
                    s1_a = ta[0:5]
                    s1_b = ta[6:]
                    if s1_a in disabled_detectors or s1_b in disabled_detectors: continue
                    if s1_a not in enabled_detectors or s1_b not in enabled_detectors: continue
                
                    '''
                    Append unique indices to S1_indices, 
                    these are the indices we want to remove.
                    '''
                    S1_indices[s1_a] = np.unique(np.append(S1_indices[s1_a], coincidence_indices[i][0]))
                    S1_indices[s1_b] = np.unique(np.append(S1_indices[s1_b], coincidence_indices[i][1]))
            
                # Remove the indices from time stamps and pulses
                if doubles_mode == 0:
                    for s1, ind in S1_indices.items():
                        time_stamps[s1] = np.delete(time_stamps[s1], S1_indices[s1])
                        pulses[s1] = np.delete(pulses[s1], S1_indices[s1], axis = 0)
                elif doubles_mode == 1:
                    for s1, ind in S1_indices.items():
                        time_stamps[s1] = time_stamps[s1][S1_indices[s1]]
                        pulses[s1] = pulses[s1][S1_indices[s1]]
                    
            print(f'\nTime range set to: {time_slice[0]:.2f} < t < {time_slice[1]:.2f}')

            # Run pulse waveform function
            print('Performing time-pickoff.')
            # Setup arguments given to create_TOF()
            argu = ['01 A '+ shot_number, '01 B '+ shot_number,  '01 C '+ shot_number,  '01 D '+ shot_number,
                    '02 A '+ shot_number, '02 B '+ shot_number,  '02 C '+ shot_number,  '02 D '+ shot_number,
                    '03 A '+ shot_number, '03 B '+ shot_number,  '03 C '+ shot_number,  '03 D '+ shot_number,
                    '04 A '+ shot_number, '04 B '+ shot_number,  '04 C '+ shot_number,  '04 D '+ shot_number,
                    '05 A '+ shot_number, '05 B '+ shot_number,  '05 C '+ shot_number,  '05 D '+ shot_number,
                    '06 A '+ shot_number, '06 B '+ shot_number,  '06 C '+ shot_number,  '06 D '+ shot_number,
                    '07 A '+ shot_number, '07 B '+ shot_number,  '07 C '+ shot_number,  '07 D '+ shot_number,
                    '08 A '+ shot_number, '08 B '+ shot_number,  '08 C '+ shot_number,  '08 D '+ shot_number,
                    '09 A '+ shot_number, '09 B '+ shot_number,  '09 C '+ shot_number,  '09 D '+ shot_number,
                    '10 B '+ shot_number, '10 C '+ shot_number,  '10 D '+ shot_number,]
            
            # Run on 16 cores
            with mp.Pool(available_cpu) as pool:
                new_times = pool.map(create_TOF,  argu)
            # Store results in dictionaries
            new_times_S1, new_times_S2 = dfs.get_dictionaries()
            energy_S1, energy_S2 = dfs.get_dictionaries()
            
            # Reorder it in ascending manner
            for i in range(0, len(new_times)):
                if   new_times[i][1] in new_times_S1: 
                    new_times_S1[new_times[i][1]] = new_times[i][0]
                    energy_S1[new_times[i][1]] = new_times[i][2]
                elif new_times[i][1] in new_times_S2: 
                    new_times_S2[new_times[i][1]] = new_times[i][0]
                    energy_S2[new_times[i][1]] = new_times[i][2]
            
            # Save data to scratch
            if not disable_scratch:
                if not scratch_exists: os.mkdir(scratch_path)
                with open(f'{scratch_path}{shot_number}.pickle', 'wb') as handle:
                    to_pickle = {'new_times_S1':new_times_S1,
                                 'new_times_S2':new_times_S2,
                                 'energy_S1':energy_S1,
                                 'energy_S2':energy_S2,
                                 'arguments':sys.argv,
                                 'time_slice':time_slice}
                    pickle.dump(to_pickle, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
        else:
            print(f'Using scratch data from {scratch_path}')
            # Get data from scratch
            with open(f'{scratch_path}{shot_number}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            new_times_S1 = data['new_times_S1']
            new_times_S2 = data['new_times_S2']
            energy_S1    = data['energy_S1']
            energy_S2    = data['energy_S2']
            time_slice   = data['time_slice']
                
        # Check if figure is already open, if so close it
        if plt.fignum_exists(shot_number): plt.close(shot_number)
    
        print('Generating TOF spectrum...')
            
        indices_S1 = np.array([])
        indices_S2 = np.array([])
        energies_S1 = np.array([])
        energies_S2 = np.array([])
        coincidences = np.array([])
                
        # Used to count empty data sets
        warn_S1 = dfs.get_dictionaries('S1')
        warn_S2 = dfs.get_dictionaries('S2')
        
        # Get shifts for aligning S1's vs S2's
        shifts = dfs.get_shifts(shift_file, time_level)
        
        # Merge new time stamps
        new_times = {**new_times_S1, **new_times_S2}
        # Arguments for parallelization of finding coincidences between S1's and S2's                
        tof_argu = [[f'{s1} {s2}', new_times] for s1 in s1_dicts.keys() for s2 in s2_dicts.keys()]
        
        # Dictionary to store coincidences in
        coincidences_raw = dfs.get_dictionaries('nested')
        coincidences_new = dfs.get_dictionaries('nested')
        indices_raw = dfs.get_dictionaries('nested')
        indices = dfs.get_dictionaries('nested')
        energies = dfs.get_dictionaries('nested')
        with mp.Pool(available_cpu) as pool_tof:
            tof_info = pool_tof.map(get_tof, tof_argu)
        
        # Store results in dictionaries
        for ti in tof_info:
            s1 = ti[2]
            s2 = ti[3]
            if s1 in disabled_detectors or s2 in disabled_detectors: continue
            if s1 not in enabled_detectors or s2 not in enabled_detectors: continue
            # Shift and store coincidences, indices and energies
            coincidences_new[s1][s2] = ti[0] + shifts[s1][int(s2[3:]) - 1]
            indices[s1][s2] = np.array([ti[1][0], ti[1][1]])
            
            # Make 1D array with tof and energies for plotting
            if not plot_1D: 
                energies[s1][s2] = np.array([energy_S1[s1][indices[s1][s2][0]], energy_S2[s2][indices[s1][s2][1]]])
                energies_S1 = np.append(energies_S1, energies[s1][s2][0])
                energies_S2 = np.append(energies_S2, energies[s1][s2][1])
            coincidences = np.append(coincidences, coincidences_new[s1][s2])
            
            
            # Warn about empty data sets
            if np.shape(new_times_S1[s1]) == (): warn_S1[s1] = 1
            if np.shape(new_times_S2[s2]) == (): warn_S2[s2] = 1
            
        # Warn about missing data
        warn_1 = 0
        warn_2 = 0
        for s1 in warn_S1: 
            if warn_S1[s1]: warn_1 += 1
        for s2 in warn_S2: 
            if warn_S2[s2]: warn_2 += 1
        
        if warn_1 or warn_2: print('WARNING: Data on ' + str(warn_1 + warn_2) + ' channels is missing for this shot.')
    
        
        # Cut out times below -150 ns and above 250 ns for plotting
        cut_low  = -150
        cut_high = 250
        cut_inds = np.where((coincidences < cut_high) & (coincidences > cut_low))[0]
        coincidences = coincidences[cut_inds]
        if not plot_1D:
            energies_S1 = energies_S1[cut_inds]
            energies_S2 = energies_S2[cut_inds]
        
        # Perform kinematic cuts
        if not disable_cuts: 
            coincidences_cut, energies_S1_cut, energies_S2_cut = dfs.kinematic_cuts(coincidences, energies_S1, energies_S2, timer = True)
        else:
            coincidences_cut = 0
            energies_S1_cut  = 0
            energies_S2_cut  = 0
            
        dfs.elapsed_time(t_start, 'all boards')
        
        if not plot_1D:
            # Plot 2D spectrum
            tof_hist, erg_S1_hist, erg_S2_hist, hist2d_S1, hist2d_S2 = dfs.plot_2D(times_of_flight = coincidences, 
                         bins_tof = bins, energy_S1 = energies_S1, energy_S2 = energies_S2, 
                         bins_energy = bins_energy, bins_2D = [bins, bins_energy], 
                         energy_lim = np.array([E_low, E_high]), interactive_plot = interactive_plot, 
                         title = f'#{shot_number} {time_slice[0]:.1f}-{time_slice[1]:.1f} s', 
                         disable_cuts = disable_cuts, energy_S1_cut = energies_S1_cut, 
                         energy_S2_cut = energies_S2_cut, times_of_flight_cut = coincidences_cut, 
                         disable_bgs = disable_bgs, sum_shots = sum_shots, proton_recoil = proton_recoil, timer = time_level)
            tof_vals    += tof_hist[0]
            erg_S1_vals += erg_S1_hist[0]
            erg_S2_vals += erg_S2_hist[0]
            
            # If final loop for summed shots, plot 2D spectrum unless data is to be saved
            if len(shots) > 1 and counter == len(shots) - 1 and not save_NES:
                dfs.plot_2D(times_of_flight = tof_vals, bins_tof = bins, energy_S1 = erg_S1_vals, 
                         energy_S2 = erg_S2_vals, bins_energy = bins_energy, bins_2D = [bins, bins_energy], 
                         energy_lim = np.array([E_low, E_high]), interactive_plot = False, title = 'Summed shots', 
                         disable_cuts = disable_cuts, energy_S1_cut = energies_S1_cut, 
                         energy_S2_cut = energies_S2_cut, times_of_flight_cut = coincidences_cut, 
                         hist2D_S1 = hist2d_S1[0], hist2D_S2 = hist2d_S2[0],
                         disable_bgs = disable_bgs, weights = True, proton_recoil = proton_recoil, timer = time_level)
            else: plt.close('all')
        else:
            # Create histogram
            dfs.hist_1D_s(coincidences, bins = np.arange(cut_low, cut_high, 0.4), 
                          y_label = 'Counts', x_label = 'Time [ns]', normed = False)
            plt.show()
            
    
        # Save all times, energies and times of flight
        if save_data:
            print(f'Saving data to: {file_name}')
            with open(file_name, 'wb') as handle:
                to_pickle = {'times_of_flight':coincidences_new, 
                             'energies':energies,
                             'times_S1':new_times_S1,
                             'times_S2':new_times_S2}
                pickle.dump(to_pickle, handle, protocol = pickle.HIGHEST_PROTOCOL)
        # Only save histogram data
        if save_NES:
            print(f'Saving NES data to {filename_NES}')
            with open(filename_NES, 'wb') as handle:
                counts, _ = np.histogram(coincidences, bins = bins)
                bin_centres = bins[0:-1] + np.diff(bins)[0] / 2
                start = np.searchsorted(bin_centres, -100)
                stop = np.searchsorted(bin_centres, -50)
                background = np.mean(tof_vals[start:stop])
                erg_bin_centres = erg_S1_hist[1][:-1] + np.diff(erg_S1_hist[1])[0] / 2 
                processed_shots = np.append(processed_shots, shot_number)
                to_pickle = {'bins':bin_centres,
                             'counts':tof_vals,
                             'bgr_level':background,
                             'erg_S1':erg_S1_vals,
                             'erg_S2':erg_S2_vals,
                             'erg_bins':erg_bin_centres,
                             'shots':processed_shots}
                pickle.dump(to_pickle, handle, protocol = pickle.HIGHEST_PROTOCOL)
            

         
