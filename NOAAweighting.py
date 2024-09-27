# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:46:12 2024

@author: kaity
@ Origion al author Jakob Tougard 
https://www.sciencedirect.com/science/article/pii/S0003682X18303967#s0095
"""

import numpy as np
import numpy as np

def NOAAweighted(audio_data, fs, filtertype):
    """
    Filters input signal sig according to recommendations from NOAA/NMFS.
    
    Parameters:
    sig : numpy array
        Input signal to be filtered.
    fs : float
        Sample rate of the input signal (in Hz).
    filtertype : str
        Type of filter:
        'HF'      : HF-cetacean
        'MF'      : MF-cetacean
        'LF'      : LF-cetacean
        'Otariid' : Otariids
        'Phocid'  : Phocids
    
    Returns:
    s : numpy array
        Filtered signal.
    
    Raises:
    ValueError : If filter type is not recognized.
    """
    
    if audio_data.ndim == 1:
        audio_data = audio_data[:, np.newaxis]  # Ensure sig is a column vector if it's 1D
    
    filtertype_upper = filtertype.upper()
    
    # Set filter parameters based on filtertype
    if filtertype_upper == 'HF':
        a, b = 1.8, 2
        f1, f2 = 12000, 140000
        C = 1.36
    elif filtertype_upper == 'MF':
        a, b = 1.6, 2
        f1, f2 = 8800, 110000
        C = 1.3
    elif filtertype_upper == 'LF':
        a, b = 1, 2
        f1, f2 = 200, 19000
        C = 0.13
    elif filtertype_upper == 'OTARIID':
        a, b = 1, 2
        f1, f2 = 1900, 30000
        C = 0.75
    elif filtertype_upper == 'PHOCID':
        a, b = 2, 2
        f1, f2 = 940, 45000
        C = 0.64
    else:
        raise ValueError('Filter type not recognized')
    
    L = audio_data.shape[0]  # Length of the input signal
    freq_axis = np.fft.fftfreq(L, 1/fs)  # Frequency axis
    
    # Calculate the weighting function in the frequency domain
    W = np.sqrt((freq_axis/f1)**(2*a) / ((1 + (freq_axis/f1)**2)**a * (1 + (freq_axis/f2)**2)**b) * 10**(C/20))
    
    # Make sure the weighting function has the correct symmetry for convol
    W = np.concatenate((W, W[::-1]))  # Two-sided spectrum with zero phase
    
    # Apply the weighting in the frequency domain
    sig_fft = np.fft.fft(audio_data, axis=0)
    s_fft = W * sig_fft
    s = np.real(np.fft.ifft(s_fft, axis=0))  # Convert back to time domain
    
    return s.squeeze()  # Return as a 1D array if input was 1D
    # If input was a horizontal array, then flip back to horizontal before returning



from scipy.signal import butter, filtfilt

def NOAAweighted(audio_data, fs, filtertype):
    """
    Filters input signal sig according to recommendations from NOAA/NMFS.
    
    Parameters:
    sig : numpy array
        Input signal to be filtered.
    fs : float
        Sample rate of the input signal (in Hz).
    filtertype : str
        Type of filter:
        'HF'      : HF-cetacean
        'MF'      : MF-cetacean
        'LF'      : LF-cetacean
        'Otariid' : Otariids
        'Phocid'  : Phocids
    
    Returns:
    s : numpy array
        Filtered signal.
    
    Raises:
    ValueError : If filter type is not recognized.
    """
    if filtertype.upper() == 'HF':
        f1, f2 = 12000, 140000
    elif filtertype.upper() == 'MF':
        f1, f2 = 8800, 110000
    elif filtertype.upper() == 'LF':
        f1, f2 = 200, 19000
    elif filtertype.upper() == 'OTARIID':
        f1, f2 = 1900, 30000
    elif filtertype.upper() == 'PHOCID':
        f1, f2 = 940, 45000
    else:
        raise ValueError('Filter type not recognized')
    
    # Normalize critical frequencies to Nyquist frequency
    nyquist = 0.5 * fs
    low = 1000 / nyquist
    high = np.min([f2, (fs-1)/2]) / nyquist
    
    # Ensure the critical frequencies are within the valid range (0, 1)
    if low <= 0 or high >= 1 or low >= high:
        raise ValueError('Invalid critical frequencies. Ensure 0 < low < high < 1.')
    
    # Design a Butterworth bandpass filter
    order = 6  # Filter order (adjust as needed)
    b, a = butter(4, low, btype='high')
    
    # Apply the filter to the signal
    s = filtfilt(b, a, audio_data)    
    
    s = s.squeeze() 
    return s.squeeze()  # Return as a 1D array if input was 1D




# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    
    # Generate example signal (replace with your own data)
    fs = 16000  # Sample rate
    t = np.linspace(0, 1, fs*2, endpoint=False)
    signal = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)
    
    # Apply NOAAweighted filter
    filtered_signal = NOAAweighted(signal, fs, 'HF')
    
    # Plotting example
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t, filtered_signal, label='Filtered Signal (HF)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Filtered Signal using NOAAweighted')
    plt.legend()
    plt.tight_layout()
    plt.show()
