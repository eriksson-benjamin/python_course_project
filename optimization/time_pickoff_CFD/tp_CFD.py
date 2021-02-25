import numpy as np
import subtract_values

@profile
def linear_regression(x_data, y_data, timer = False):
    '''
    Returns the slope (A) and intersection (B) for a simple linear regression on x and y data.
    x_data: 2D array of values where each row corresponds to one event to perform linear regression on
    y_data: 2D array of values where each row corresponds to one event to perform linear regression on
    product_1, product_2 and product_3 correspond to the three products for calculating beta in 
    https://en.wikipedia.org/wiki/Simple_linear_regression
    '''
#    if timer: t_start = elapsed_time()
    
    # Find average
    x_mean = np.mean(x_data, axis = 1)
    y_mean = np.mean(y_data, axis = 1)
    
    product_1 = np.transpose(np.transpose(x_data) - x_mean)
    product_2 = np.transpose(np.transpose(y_data) - y_mean)
    product_3 = product_1 ** 2
    
    # Calculate slopes and intersection (y = slope*x + intercept)
    slope = np.sum(product_1 * product_2, axis = 1) / np.sum(product_3, axis = 1)
    intercept = np.mean(y_data, axis = 1) - slope * x_mean
    
#    if timer: elapsed_time(t_start, 'linear_regression()')    
    return slope, intercept

#@profile
#def find_points(pulse_data, value, timer = False):
#    '''
#    Returns the index of the point closest to "value" in pulse_data.
#    pulse_data: array of pulse height data where each row corresponds to one record. 
#                NOTE: pulse_data must be baseline reduced (see baseline_reduction() function).
#    value: one dimensional array of values for which you want to find the closest index in pulse_data 
#    '''
##    if timer: t_start = elapsed_time()
#    
#    # Subtract the constant fraction value from the data set
##    delta = pulse_data - value[:, None]
#    delta = np.subtract(pulse_data, value[:, np.newaxis])
#    # Find the index of the first positive value
#    mask = delta <= 0
#    
#    index = np.argmax(mask, axis = 1) 
#    
##    if timer: elapsed_time(t_start, 'find_points()')
#    return index  

@profile
def find_points(pulse_data, value, timer = False):
    '''
    Returns the index of the point closest to "value" in pulse_data.
    pulse_data: array of pulse height data where each row corresponds to one record. 
                NOTE: pulse_data must be baseline reduced (see baseline_reduction() function).
    value: one dimensional array of values for which you want to find the closest index in pulse_data 
    '''
#    if timer: t_start = elapsed_time()
    
    # Subtract the constant fraction value from the data set

    mask = subtract_values.subtract_values(pulse_data, value[:, np.newaxis])
    index = np.argmax(mask, axis = 1) 
    
#    if timer: elapsed_time(t_start, 'find_points()')
    return index  


@profile
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

#    if timer: t_start = elapsed_time()
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
        print(type(minima_fraction[0]))
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
        

#    if timer: elapsed_time(t_start, 'time_pickoff_CFD()')
    return new_time
