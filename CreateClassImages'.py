# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py




################# NO TOUCHIE##################################
def load_and_process_audio_segment(file_path, start_time, end_time,
                                      clipDur =2, outSR = 16000):
    """
    Load an audio segment from a file, process it as described, and create a spectrogram image.
    
    Parameters:
        file_path (str): Path to the audio file.
        start_time (float): Start time of the audio segment in seconds.
        end_time (float): End time of the audio segment in seconds.
    
    Returns:
        spec_img (matplotlib.image.AxesImage): Spectrogram image of the downsampled audio segment.
    """
    # Set the duration to 2 seconds
    duration = clipDur
    
    # Calculate the start and end frame based on the provided start and end time
    sample_rate = librosa.get_samplerate(file_path)
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)
    
    # Adjust start and end frames to ensure the segment is 2 seconds
    if end_frame - start_frame < sample_rate * duration:
        mid_frame = (start_frame + end_frame) // 2
        start_frame = mid_frame - int(sample_rate * duration / 2)
        end_frame = mid_frame + int(sample_rate * duration / 2)
    
    # Load audio segment
    audio_data, _ = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
    
    # Downsample if sample rate is greater than 8000
    if sample_rate > outSR:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=outSR)
        sample_rate = outSR
    
    # Create spectrogram
    n_fft = 512
    hop_length = int(n_fft * .2)  # 90% overlap
    #spec = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
    
    
    spec = librosa.feature.melspectrogram(y=audio_data,
                                          sr=sample_rate,
                                          n_fft=n_fft,
                                          hop_length=hop_length)
    spec = librosa.power_to_db(spec, ref=np.max)
    
    
    
    # Normalize spectrogram KISS
    row_medians = np.median(spec, axis=1, keepdims=True)
    col_medians = np.median(spec, axis=0, keepdims=True)
    spec_normalized = spec - row_medians - col_medians
    
    spec_img = librosa.display.specshow(spec_normalized,
                                          sr=sample_rate,
                                          hop_length=hop_length,     
                                          x_axis='time',
                                          y_axis='log')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    plt.tight_layout()
    plt.show()
    
    return spec_normalized




def load_and_process_audio_segment(file_path, start_time, end_time,
                                      clipDur=2, outSR=16000):
    """
    Load an audio segment from a file, process it as described, and create a spectrogram image.
    
    Parameters:
        file_path (str): Path to the audio file.
        start_time (float): Start time of the audio segment in seconds.
        end_time (float): End time of the audio segment in seconds.
        clipDur (float): Duration of the desired spectrogram in seconds.
        outSR (int): Target sample rate for resampling.
    
    Returns:
        spec_normalized (numpy.ndarray): Normalized spectrogram of the audio segment.
    """
    # Get the duration of the audio file
    file_duration = librosa.get_duration(path=file_path)
    
    # Calculate the duration of the audio segment
    duration = end_time - start_time
    
    # Center time
    center_time = start_time+ duration/2
    
    # Adjust start and end times if the duration is less than clipDur
    if duration < clipDur:
        # Calculate the amount of time to add/subtract from start/end times
        time_difference = clipDur - duration
        time_to_add = time_difference / 2
        time_to_subtract = time_difference - time_to_add
        
        # Adjust start and end times
        start_time = max(0, start_time - time_to_subtract)
        end_time = min(end_time + time_to_add, file_duration)
        #warnings.warn(f"Adjusted start time to {start_time} and end time to {end_time} to ensure spectrogram size consistency.")
    
    # Ensure start and end times don't exceed the bounds of the audio file
    start_time = max(0, min(start_time, file_duration - clipDur))
    end_time = max(clipDur, min(end_time, file_duration))
    
    # Load audio segment
    audio_data, sample_rate = librosa.load(file_path, sr=outSR, offset=start_time, duration=clipDur)
    
    # Create spectrogram
    n_fft = 512
    hop_length = int(n_fft * 0.2)  # 90% overlap
    
    spec = librosa.feature.melspectrogram(y=audio_data,
                                          sr=outSR,
                                          n_fft=n_fft,
                                          hop_length=hop_length)
    spec = librosa.power_to_db(spec, ref=np.max)
    
    # Normalize spectrogram
    row_medians = np.median(spec, axis=1, keepdims=True)
    col_medians = np.median(spec, axis=0, keepdims=True)
    spec_normalized = spec - row_medians - col_medians
    
    # Get SNR using 25th and 85th percentiles
    sig = np.quantile(spec_normalized, np.array([0.85]))
    noise = np.quantile(spec_normalized, np.array([0.25]))
    SNR = np.round(sig-noise,1)
    
    return spec_normalized, SNR



def load_and_process_audio_segment(file_path, start_time, end_time,
                                   clipDur=2, outSR=16000):
    """
    Load an audio segment from a file, process it as described, and create a spectrogram image.
    
    Parameters:
        file_path (str): Path to the audio file.
        start_time (float): Start time of the audio segment in seconds.
        end_time (float): End time of the audio segment in seconds.
        clipDur (float): Duration of the desired spectrogram in seconds.
        outSR (int): Target sample rate for resampling.
    
    Returns:
        spec_normalized (numpy.ndarray): Normalized spectrogram of the audio segment.
        SNR (float): Signal-to-Noise Ratio of the spectrogram.
    """
    # Get the duration of the audio file
    file_duration = librosa.get_duration(path=file_path)
    
    # Calculate the duration of the audio segment
    duration = end_time - start_time
    
    # Calculate the center time of the desired clip
    center_time = (start_time + end_time) / 2.0
    
    # Calculate new start and end times based on the center and clip duration
    new_start_time = center_time - clipDur / 2
    new_end_time = center_time + clipDur / 2
    
    # Adjust start and end times if the clip duration is less than desired
    if new_end_time - new_start_time < clipDur:
        pad_length = clipDur - (new_end_time - new_start_time)
        new_start_time = max(0, new_start_time - pad_length / 2.0)
        new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
    
    # Ensure start and end times don't exceed the bounds of the audio file
    new_start_time = max(0, min(new_start_time, file_duration - clipDur))
    new_end_time = max(clipDur, min(new_end_time, file_duration))
    
    # Load audio segment
    audio_data, sample_rate = librosa.load(file_path, sr=outSR, offset=new_start_time, duration=clipDur)
    
    # Create spectrogram
    n_fft = 512
    hop_length = int(n_fft * 0.2)  # 90% overlap
    
    spec = librosa.feature.melspectrogram(y=audio_data,
                                          sr=outSR,
                                          n_fft=n_fft,
                                          hop_length=hop_length)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    # Normalize spectrogram
    row_medians = np.median(spec_db, axis=1, keepdims=True)
    col_medians = np.median(spec_db, axis=0, keepdims=True)
    spec_normalized = spec_db - row_medians - col_medians
    
    # Calculate SNR using 25th and 85th percentiles
    signal_level = np.percentile(spec_normalized, 85)
    noise_level = np.percentile(spec_normalized, 25)
    SNR = signal_level - noise_level
    
    return spec_normalized, float(SNR)

# Load and process audio segments, and save spectrograms and labels to HDF5 file
def create_hdf5_dataset(annotations, hdf5_filename):
    with h5py.File(hdf5_filename, 'w') as hf:
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        n_rows = len(annotations)
        # Read CSV file and process data
        # Replace this with your code to iterate through the CSV file and generate spectrogram clips
        #for index, row in AllAnno.iterrows():
        for ii in range(0, n_rows):
            row = annotations.iloc[ii]
            file_path = row['FilePath']
            start_time = row['FileBeginSec']
            end_time = row['FileEndSec']
            label = row['label']
            traintest = row['traintest']
            dep = row['Dep']
            provider = row['Provider']
            kw = row['KW']
            kwCertin = row['KW_certain']

            spec, SNR = load_and_process_audio_segment(file_path, start_time, end_time)

            if traintest == 'Train':  # 80% train, 20% test
                dataset = train_group
            else:
                dataset = test_group

            dataset.create_dataset(f'{ii}/spectrogram', data=spec)
            dataset.create_dataset(f'{ii}/label', data=label)
            dataset.create_dataset(f'{ii}/deployment', data=dep)
            dataset.create_dataset(f'{ii}/provider', data=provider)
            dataset.create_dataset(f'{ii}/KW', data=kw)
            dataset.create_dataset(f'{ii}/KW_certain', data=kwCertin)
            dataset.create_dataset(f'{ii}/SNR', data=SNR)
            
            
            print(ii, ' of ', len(annotations))






def check_spectrogram_dimensions(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hf:
        spectrogram_shapes = set()  # Set to store unique spectrogram shapes
        for group_name in hf:
            group = hf[group_name]
            for key in group:
                spectrograms = group[key]['spectrogram'][:]
                for spectrogram in spectrograms:
                    spectrogram_shapes.add(spectrogram.shape)
    return spectrogram_shapes

# Usage example
# Example usage within the same script
if __name__ == "__main__":
    
    
    annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain.csv")
    annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest.csv")
    
    AllAnno = pd.concat([annot_train, annot_val], axis=0)
    AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]
    AllAnno = AllAnno.dropna(subset=['FileEndSec'])
    
    
    #AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)
    annot_malahat = pd.read_csv('C:/Users/kaity/Documents/GitHub/Ecotype/Malahat.csv')
    Validation =annot_malahat[annot_malahat['LowFreqHz'] < 8000]
    
    train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/TrainTest_08_khz_Melint16MetricsSNR.h5'
    val_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatVal_08_khz_Melint16MetricsSNR.h5'
    val_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/test.h5'

    create_hdf5_dataset(annotations=AllAnno.iloc[1:20], hdf5_filename= val_hdf5_file)

    hf = h5py.File(train_hdf5_file, 'r')

    
    
    label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()
    
    label_mapping_Validation = Validation[['label', 'Labels']].drop_duplicates()
    