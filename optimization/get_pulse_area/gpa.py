import numpy as np

@profile
def get_pulse_area(pulses, u_factor):
    '''
    Returns the areas under an array of pulses
    pulses: m*n array of pulses
    u_factor: frequency of samples in each pulse (u_factor = 10 -> 1/10 ns between each sample)
    '''
    
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
        
    return pulse_area