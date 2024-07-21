# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:58:40 2024

@author: kaity
"""


# Functions for creating the HDF5 databases


################# NO TOUCHIE##################################




import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Activation
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import keras
from keras import Model
from keras.callbacks import EarlyStopping


def create_spectrogram(audio, return_snr=False,**kwargs):
    """
    Create the audio representation.

    kwargs options:
        clipDur (float): clip duration (sec).
        nfft (int): FFT window size (samples).
        hop_length (int): Hop length (samples).
        outSR (int): Target sampling rate.
        spec_type (str): Spectrogram type, 'normal' or 'mel'.
        min_freq (int): Minimum frequency to retain.
        rowNorm (bool): Normalize the spectrogram rows.
        colNorm (bool): Normalize the spectrogram columns.
        rmDCoffset (bool): Remove DC offset by subtracting mean.
        inSR (int): Original sample rate of the audio file (if decimation is needed).

    Returns:
        spectrogram (numpy.ndarray): Normalized spectrogram of the audio segment.
    """
    
    # Default parameters
    params = {
        'clipDur': 2,           # clip duration (sec)
        'nfft': 512,            # default FFT window size (samples)
        'hop_length': 3200,     # default hop length (samples)
        'outSR': 16000,         # default target sampling rate
        'spec_type': 'normal',  # default spectrogram type, 'normal' or 'mel'
        'min_freq': None,       # default minimum frequency to retain
        'rowNorm': True,        # normalize the spectrogram rows
        'colNorm': True,        # normalize the spectrogram columns
        'rmDCoffset': True,     # remove DC offset by subtracting mean
        'inSR': None,            # original sample rate of the audio file
    }

    # Update parameters based on kwargs
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            raise TypeError(f"Invalid keyword argument '{key}'")
    
    # Downsample data if necessary
    if params['inSR'] and params['inSR'] != params['outSR']:
        audio = librosa.resample(audio, orig_sr=params['inSR'], target_sr=params['outSR'])
    
    # Remove DC offset
    if params['rmDCoffset']:
        audio = audio - np.mean(audio)
    
    # Compute spectrogram
    if params['spec_type'] == 'normal':
        spectrogram = np.abs(librosa.stft(audio, n_fft=params['nfft'], hop_length=params['hop_length']))
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
    elif params['spec_type'] == 'mel':
        
        spectrogram = librosa.feature.melspectrogram(y=audio, 
                                                     sr=params['outSR'], 
                                                     n_fft=params['nfft'], 
                                                     hop_length=params['hop_length'])
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
    else:
        raise ValueError("Invalid spectrogram type. Supported types are 'normal' and 'mel'.")
    
    # Normalize the spectrogram
    if params['rowNorm']:
        row_medians = np.median(spectrogram, axis=1, keepdims=True)
        spectrogram = spectrogram - row_medians
        
    if params['colNorm']:
        col_medians = np.median(spectrogram, axis=0, keepdims=True)
        spectrogram = spectrogram - col_medians
    
    # Trim frequencies if min_freq is specified
    if params['min_freq'] is not None:
        mel_frequencies = librosa.core.mel_frequencies(n_mels=spectrogram.shape[0] + 2)
        min_idx = np.argmax(mel_frequencies >= params['min_freq'])
        spectrogram = spectrogram[min_idx:, :]
        
    # Calculate SNR if requested
    if return_snr:
        signal_level = np.percentile(spectrogram, 85)
        noise_level = np.percentile(spectrogram, 25)
        SNR = signal_level - noise_level
        return spectrogram, SNR
    
    return spectrogram

# Redefine to independnetly create the audio representations
def load_and_process_audio_segment(file_path, start_time, end_time,
                                   return_snr=False, **kwargs):
    """
    Load an audio segment from a file, process it, and create a spectrogram image.

    Parameters:
        file_path (str): Path to the audio file.
        start_time (float): Start time of the audio segment in seconds.
        end_time (float): End time of the audio segment in seconds.
        return_snr (bool): Flag to return Signal-to-Noise Ratio of the spectrogram.
        **kwargs: Additional keyword arguments passed to create_spectrogram.

    Returns:
        spectrogram (numpy.ndarray): Normalized spectrogram of the audio segment.
        SNR (float): Signal-to-Noise Ratio of the spectrogram (if return_snr=True).
    """
    
    # Default parameters for create_spectrogram
    params = {
        'clipDur': 2,           # clip duration (sec)
        'nfft': 512,            # default FFT window size (samples)
        'hop_length': 3200,     # default hop length (samples)
        'outSR': 16000,         # default target sampling rate
        'spec_type': 'normal',  # default spectrogram type, 'normal' or 'mel'
        'min_freq': None,       # default minimum frequency to retain
        'rowNorm': True,        # normalize the spectrogram rows
        'colNorm': True,        # normalize the spectrogram columns
        'rmDCoffset': True,     # remove DC offset by subtracting mean
        'inSR': None,           # original sample rate of the audio file
    }

    # Update parameters based on kwargs
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            raise TypeError(f"Invalid keyword argument '{key}'")

    # Get the duration of the audio file
    file_duration = librosa.get_duration(filename=file_path)
    
    # Calculate the duration of the desired audio segment
    duration = end_time - start_time
    
    # Calculate the center time of the desired clip
    center_time = (start_time + end_time) / 2.0
    
    # Calculate new start and end times based on the center and clip duration
    new_start_time = center_time - params['clipDur'] / 2
    new_end_time = center_time + params['clipDur'] / 2
    
    # Adjust start and end times if the clip duration is less than desired
    if new_end_time - new_start_time < params['clipDur']:
        pad_length = params['clipDur'] - (new_end_time - new_start_time)
        new_start_time = max(0, new_start_time - pad_length / 2.0)
        new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
    
    # Ensure start and end times don't exceed the bounds of the audio file
    new_start_time = max(0, min(new_start_time, file_duration - params['clipDur']))
    new_end_time = max(params['clipDur'], min(new_end_time, file_duration))
    
    # Load audio segment and downsample to the defined sampling rate
    audio_data, sample_rate = librosa.load(file_path, sr=params['outSR'], 
                                           offset=new_start_time,
                                           duration=params['clipDur'])

    # Create audio representation
    spec, snr = create_spectrogram(audio_data, return_snr=return_snr, **params)

    if return_snr:
        return spec, snr
    else:
        return spec





# def load_and_process_audio_segment(file_path, start_time, end_time, **kwargs
#                                    clipDur=2, outSR=16000, y= None):
#     """
#     Load an audio segment from a file, process it as described, and create a spectrogram image.
    
#     Parameters:
#         file_path (str): Path to the audio file.
#         start_time (float): Start time of the audio segment in seconds.
#         end_time (float): End time of the audio segment in seconds.
#         clipDur (float): Duration of the desired spectrogram in seconds.
#         outSR (int): Target sample rate for resampling.
    
#     Returns:
#         spec_normalized (numpy.ndarray): Normalized spectrogram of the audio segment.
#         SNR (float): Signal-to-Noise Ratio of the spectrogram.
#     """
    
#     # Pull out the parameters
#     # Update parameters based on kwargs
#     for key, value in kwargs.items():
#         if key in params:
#             params[key] = value
#         else:
#             raise TypeError(f"Invalid keyword argument '{key}'")
    
    
#     outSR = params['outSR']
#     clipDur = params['clipDur']
    
#     # trying to reuse this for the detector
#     if isinstance(file_path, str):
#     # Get the duration of the audio file
#         file_duration = librosa.get_duration(path=file_path)
        
#         # Calculate the duration of the audio segment
#         duration = end_time - start_time
        
#         # Calculate the center time of the desired clip
#         center_time = (start_time + end_time) / 2.0
        
#         # Calculate new start and end times based on the center and clip duration
#         new_start_time = center_time - clipDur / 2
#         new_end_time = center_time + clipDur / 2
        
#         # Adjust start and end times if the clip duration is less than desired
#         if new_end_time - new_start_time < clipDur:
#             pad_length = clipDur - (new_end_time - new_start_time)
#             new_start_time = max(0, new_start_time - pad_length / 2.0)
#             new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
        
#         # Ensure start and end times don't exceed the bounds of the audio file
#         new_start_time = max(0, min(new_start_time, file_duration - clipDur))
#         new_end_time = max(clipDur, min(new_end_time, file_duration))
        
#         # Load audio segment and downsample to the defined FS
#         audio_data, sample_rate = librosa.load(file_path, sr=outSR, offset=new_start_time, duration=clipDur)
    
#     # data already loaded
#     else:
#         audio_data= file_path
        
    
#     # Create spectrogram
#     n_fft = 512
#     hop_length = int(n_fft * 0.2)  # 90% overlap
    
#     spec = librosa.feature.melspectrogram(y=audio_data,
#                                           sr=outSR,
#                                           n_fft=n_fft,
#                                           hop_length=hop_length)
#     spec_db = librosa.power_to_db(spec, ref=np.max)
    
#     # Normalize spectrogram
#     row_medians = np.median(spec_db, axis=1, keepdims=True)
#     col_medians = np.median(spec_db, axis=0, keepdims=True)
#     spec_normalized = spec_db - row_medians - col_medians
    
#     # Calculate SNR using 25th and 85th percentiles
#     signal_level = np.percentile(spec_normalized, 85)
#     noise_level = np.percentile(spec_normalized, 25)
#     SNR = signal_level - noise_level
    
#     return spec_normalized, float(SNR)

# Load and process audio segments, and save spectrograms and labels to HDF5 file
def create_hdf5_dataset(annotations, hdf5_filename, parms):
    """
    Create an HDF5 database with spectrogram images for DCLDE data.
    
    Parameters:
        annotations (pd.df): pandas dataframe with headers for 'FilePath', 
        'FileBeginSec','FileEndSec', 'label','traintest', 'Dep', 'Provider',
        'KW','KW_certain'
    Returns:
        spec_normalized (numpy.ndarray): Normalized spectrogram of the audio segment.
        SNR (float): Signal-to-Noise Ratio of the spectrogram.
    """
    # write the parameters for recovering latter
    
    with h5py.File(hdf5_filename, 'w') as hf:
        parms = hf.create_dataset('AudioParameters', data=parms)
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
            utc = row['UTC']

            spec, SNR = load_and_process_audio_segment(file_path, start_time, end_time,
                                               return_snr=True, **parms)

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
            dataset.create_dataset(f'{ii}/UTC', data=utc)
            
            
            print(ii, ' of ', len(annotations))




import h5py
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define your functions load_and_process_audio_segment() and create_hdf5_dataset() here

# Function to process each row in annotations DataFrame
def process_row(row):
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

    return spec, SNR, traintest, label, dep, provider, kw, kwCertin

# Main function to create HDF5 dataset with multithreading
def create_hdf5_dataset_parallel(annotations, hdf5_filename, num_threads=4):
    with h5py.File(hdf5_filename, 'w') as hf:
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        n_rows = len(annotations)

        # Use ThreadPoolExecutor to parallelize the processing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_row, annotations.iloc[ii]) for ii in range(n_rows)]

            for future in as_completed(futures):
                spec, SNR, traintest, label, dep, provider, kw, kwCertin = future.result()

                if traintest == 'Train':  # 80% train, 20% test
                    dataset = train_group
                else:
                    dataset = test_group

                ii = len(dataset)  # Get the index for the new dataset

                dataset.create_dataset(f'{ii}/spectrogram', data=spec)
                dataset.create_dataset(f'{ii}/label', data=label)
                dataset.create_dataset(f'{ii}/deployment', data=dep)
                dataset.create_dataset(f'{ii}/provider', data=provider)
                dataset.create_dataset(f'{ii}/KW', data=kw)
                dataset.create_dataset(f'{ii}/KW_certain', data=kwCertin)
                dataset.create_dataset(f'{ii}/SNR', data=SNR)

                print(ii, ' of ', len(annotations))



import math
    
class BatchLoader2(keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size=250, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False,
                 minFreq = 'nan'):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels=return_data_labels
        self.data_keys = list(self.hf[trainTest].keys())
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
        
       
        # Get th spectrogram size, assume something in Train
        self.train_group = self.hf[trainTest]
        self.first_key = list(self.train_group.keys())[0]
        self.data =self.train_group[self.first_key]['spectrogram']
        self.specSize = self.data.shape
        
        # If a frequency limit is set then  figure out what that is now
        self.minFreq = minFreq
        self.minIdx = 0
        
        #This is for trimming the frequency range
        if math.isfinite(self.minFreq):
             mel_frequencies = librosa.core.mel_frequencies(n_mels= self.specSize[0]+2)

             # Find the index corresponding to 500 Hz in mel_frequencies
             self.minIdx = np.argmax(mel_frequencies >= self.minFreq)
            
            # # # also update the spectrogram size
             self.data= self.data[self.minIdx:,:]
             self.specSize = self.data.shape
     
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.num_samples)
        
        batch_data = []
        batch_labels = []
        
        for i in range(start_index, end_index):
            key = self.data_keys[self.indexes[i]]
            spec = self.hf[self.trainTest][key]['spectrogram'][self.minIdx:,:]
            label = self.hf[self.trainTest][key]['label'][()]
            batch_data.append(spec)
            batch_labels.append(label)
        
        if self.return_data_labels:
            return batch_data, batch_labels
        else:
            return np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels), num_classes=self.n_classes)
    def __shuffle__(self):
        np.random.shuffle(self.indexes)
        print('shuffled!')
    
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('Epoc end all shuffled!')


import h5py
import numpy as np
import keras.utils
import math
import librosa

# class BatchLoader2pt5(keras.utils.Sequence):
#     def __init__(self, hdf5_file, batch_size=250, trainTest='train',
#                  shuffle=True, n_classes=7, return_data_labels=False,
#                  minFreq='nan', return_vars=None):
#         self.hf = h5py.File(hdf5_file, 'r')
#         self.batch_size = batch_size
#         self.trainTest = trainTest
#         self.shuffle = shuffle
#         self.n_classes = n_classes
#         self.return_data_labels = return_data_labels
#         self.return_vars = return_vars if return_vars else []
#         self.data_keys = list(self.hf[trainTest].keys())
#         self.num_samples = len(self.data_keys)
#         self.indexes = np.arange(self.num_samples)

#         # Get the spectrogram size, assuming data exists in 'train'
#         self.train_group = self.hf[trainTest]
#         self.first_key = list(self.train_group.keys())[0]
#         self.data = self.train_group[self.first_key]['spectrogram']
#         self.specSize = self.data.shape
        
#         # If a frequency limit is set, figure out what that is now
#         self.minFreq = minFreq
#         self.minIdx = 0
#         if math.isfinite(self.minFreq):
#             mel_frequencies = librosa.core.mel_frequencies(n_mels=self.specSize[0] + 2)
#             self.minIdx = np.argmax(mel_frequencies >= self.minFreq)
#             self.data = self.data[self.minIdx:, :]
#             self.specSize = self.data.shape
        
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         return int(np.ceil(self.num_samples / self.batch_size))

#     def __getitem__(self, index):
#         start_index = index * self.batch_size
#         end_index = min((index + 1) * self.batch_size, self.num_samples)
        
#         batch_data = []
#         batch_labels = []
#         batch_vars = {var: [] for var in self.return_vars}

#         for i in range(start_index, end_index):
#             key = self.data_keys[self.indexes[i]]
#             spec = self.hf[self.trainTest][key]['spectrogram'][self.minIdx:, :]
#             label = self.hf[self.trainTest][key]['label'][()]
#             batch_data.append(spec)
#             batch_labels.append(label)

#             for var in self.return_vars:
#                 if var in self.hf[self.trainTest][key]:
#                     value = self.hf[self.trainTest][key][var][()]
#                     batch_vars[var].append(value)
#                 else:
#                     raise KeyError(f"Variable '{var}' not found in HDF5 file for key '{key}'")

#         if self.return_data_labels:
#             return_list = [batch_data, batch_labels]
#             for var in self.return_vars:
#                 return_list.append(np.array(batch_vars[var]))
#             return tuple(return_list)
#         else:
#             batch_labels = keras.utils.to_categorical(np.array(batch_labels), num_classes=self.n_classes)
#             return_list = [np.array(batch_data), batch_labels]
#             for var in self.return_vars:
#                 return_list.append(np.array(batch_vars[var]))
#             return tuple(return_list)

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
#             print('Epoch end: all shuffled!')


import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class BatchLoader3:
    def __init__(self, hdf5_file, batch_size=250, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False,
                 minFreq='nan'):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels = return_data_labels
        self.data_keys = list(self.hf[trainTest].keys())
        self.num_samples = len(self.data_keys)
        
        # Get the spectrogram size from the first sample
        first_key = self.data_keys[0]
        self.data = self.hf[trainTest][first_key]['spectrogram']
        self.specSize = self.data.shape
        
        # Handle minimum frequency
        self.minFreq = minFreq
        if math.isfinite(self.minFreq):
            mel_frequencies = librosa.core.mel_frequencies(n_mels=self.specSize[0] + 2)
            self.minIdx = np.argmax(mel_frequencies >= self.minFreq)
            self.specSize = self.data[self.minIdx:, :].shape
        
        if self.shuffle:
            np.random.shuffle(self.data_keys)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def generate_dataset(self):
        # Create a list of (data, label) tuples
        dataset = []
        for key in self.data_keys:
            spec = self.hf[self.trainTest][key]['spectrogram'][self.minIdx:, :]
            label = self.hf[self.trainTest][key]['label'][()]
            dataset.append((spec, label))

        # Shuffle dataset if needed
        if self.shuffle:
            np.random.shuffle(dataset)

        # Separate data and labels
        data_list, labels_list = zip(*dataset)

        # Convert lists to numpy arrays
        data_array = np.array(data_list)
        labels_array = np.array(labels_list)

        # If returning data and labels, return as tuple
        if self.return_data_labels:
            return data_array, labels_array

        # Otherwise, return as TensorFlow Dataset
        return tf.data.Dataset.from_tensor_slices((data_array, to_categorical(labels_array, num_classes=self.n_classes)))

from tensorflow.keras import backend as K
def custom_weighted_loss(class_weights, priorities):
    """
    Custom weighted categorical crossentropy loss function.

    Parameters:
    - class_weights: Dictionary containing weights for each class.
    - priorities: List of lists where each sublist represents a group of classes with a priority.

    Returns:
    - loss function to be used in model compilation.
    """
    def loss(y_true, y_pred):
        # Convert class weights to tensor
        weights = tf.constant([class_weights[key] for key in sorted(class_weights.keys())], dtype=tf.float32)
        
        # Convert priorities to a tensor
        priority_indices = [sorted(class_weights.keys()).index(cls) for priority_group in priorities for cls in priority_group]
        priority_mask = K.one_hot(priority_indices, len(class_weights))  # Remove dtype argument
        
        # Calculate cross entropy
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clip to prevent log(0) instability
        cross_entropy = -y_true * K.log(y_pred)
        
        # Apply weights and priorities
        weighted_cross_entropy = cross_entropy * weights
        weighted_cross_entropy *= priority_mask
        
        # Sum over classes and mean over batch
        return K.mean(K.sum(weighted_cross_entropy, axis=-1))
    
    return loss

def train_model(model, train_generator, val_generator, epochs):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3, 
                                   restore_best_weights=True)
   
    # Train the model with early stopping
    model.fit(x=train_generator,  # Here, `train_generator` should be a generator object
              epochs=epochs,
              validation_data=val_generator,
              callbacks=[early_stopping])
    
    

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

###########################################################################
# Work someting up more like resnet.
#########################################################################


def create_model_with_resnet(input_shape, num_classes, actName = 'relu'):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation=actName)(input_layer)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation=actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add three identity blocks
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    
    x = Flatten()(x)
    x = Dense(128, activation=actName)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def create_wider_model(input_shape, num_classes, actName='relu'):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(3, 3), activation=actName)(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation=actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Same identity blocks
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    
    x = Flatten()(x)
    x = Dense(256, activation=actName)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def identity_block(input_tensor, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    #x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Add()([x, input_tensor])  # Add the input tensor to the output of the second convolution
    x = Activation('relu')(x)
    return x


def create_wider_model2(input_shape, num_classes, actName='relu'):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(3, 3), activation=None, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, kernel_size=(3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(actName)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Same identity blocks
    x = identity_block2(x, 64)
    x = identity_block2(x, 64)
    x = identity_block2(x, 64)
    
    x = Flatten()(x)
    x = Dense(128, activation=actName)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def identity_block2(input_tensor, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection
    shortcut = Conv2D(filters, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# def create_det_class_model_with_resnet(input_shape, num_classes_ecotype):
#     input_layer = Input(shape=input_shape)
#     x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
    
#     # Add three identity blocks
#     x = identity_block(x, 64)
#     x = identity_block(x, 64)
#     x = identity_block(x, 64)
    
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
    
#     # Output branch for killer whale detection
#     output_killer_whale = Dense(1, activation='sigmoid', name='killer_whale')(x)
    
#     # Output branch for ecotype classification
#     output_ecotype = Dense(num_classes_ecotype, activation='softmax', name='ecotype')(x)
    
#     model = Model(inputs=input_layer, outputs=[output_killer_whale, output_ecotype])
    
#     return model


# def compile_det_class_model(model):
#     model.compile(optimizer='adam',
#                   loss={'killer_whale': 'binary_crossentropy', 'ecotype': 'categorical_crossentropy'},
#                   metrics={'killer_whale': 'accuracy', 'ecotype': 'accuracy'})
#     return model



# Compile model
def compile_model(model, loss_val = 'categorical_crossentropy'):
    model.compile(loss=loss_val,
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

def compile_model_customWeights(model, class_weights, priorities):
    # Compile the model with the custom loss function
    model.compile(optimizer='adam', 
                  loss=custom_weighted_loss(class_weights, priorities),
                  metrics=['accuracy'])
    return model


from sklearn.metrics import confusion_matrix
def batch_conf_matrix(loaded_model, val_batch_loader):
    # Initialize lists to accumulate predictions and true labels
    y_pred_accum = []
    y_true_accum = []
    
    # Get the total number of batches
    total_batches = len(val_batch_loader)
    
    # Iterate over test data generator batches
    for i in range(0, len(val_batch_loader)):
        batch_data = val_batch_loader.__getitem__(i)
        
        # Predict on the current batch
        batch_pred = loaded_model.predict(batch_data[0])
        
        # Convert predictions to class labels
        batch_pred_labels = np.argmax(batch_pred, axis=1)
        
        # Convert true labels to class labels
        batch_true_labels = np.argmax(batch_data[1], axis=1)
        
        # Accumulate predictions and true labels
        y_pred_accum.extend(batch_pred_labels)
        y_true_accum.extend(batch_true_labels)
        
        # Print progress
        print(f'Batch {i+1}/{total_batches} processed')
    
    # Convert accumulated lists to arrays
    y_pred_accum = np.array(y_pred_accum)
    y_true_accum = np.array(y_true_accum)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_accum, y_pred_accum)
    
    return(conf_matrix)
    print("Confusion Matrix:")
    
# So this is well and good but I'd like to create be able to look at the timestamps
# from the predictions so we can do something lie softmax


def batch_conf_matrix(loaded_model, val_batch_loader, returnVars = list()):
    # Initialize lists to accumulate predictions and true labels
    y_pred_accum = []
    y_true_accum = []
    
    # Get the total number of batches
    total_batches = len(val_batch_loader)
    
    # Iterate over test data generator batches
    for i in range(0, len(val_batch_loader)):
        batch_data = val_batch_loader.__getitem__(i)
        
        # Predict on the current batch
        batch_pred = loaded_model.predict(batch_data[0])
        
        # Convert predictions to class labels
        batch_pred_labels = np.argmax(batch_pred, axis=1)
        
        # Convert true labels to class labels
        batch_true_labels = np.argmax(batch_data[1], axis=1)
        
        # Accumulate predictions and true labels
        y_pred_accum.extend(batch_pred_labels)
        y_true_accum.extend(batch_true_labels)
        
        # Print progress
        print(f'Batch {i+1}/{total_batches} processed')
    
    # Convert accumulated lists to arrays
    y_pred_accum = np.array(y_pred_accum)
    y_true_accum = np.array(y_true_accum)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_accum, y_pred_accum)
    
    return(conf_matrix)
    print("Confusion Matrix:")
    


###################################################################
# Model evaluation
###################################################################


# def confuseionMat(model, val_batch_loader):
#     # Initialize lists to accumulate predictions and true labels
#     y_pred_accum = []
#     y_true_accum = []
    
#     # Get the total number of batches
#     total_batches = len(val_batch_loader)
    
    
#     # Iterate over test data generator batches
#     for i, (batch_data, batch_labels) in enumerate(val_batch_loader):
#         # Predict on the current batch
#         batch_pred = model.predict(batch_data)
        
#         # Convert predictions to class labels
#         batch_pred_labels = np.argmax(batch_pred, axis=1)
        
#         # Convert true labels to class labels
#         batch_true_labels = np.argmax(batch_labels, axis=1)
        
#         # Accumulate predictions and true labels
#         y_pred_accum.extend(batch_pred_labels)
#         y_true_accum.extend(batch_true_labels)
        
#         # Print progress
#         print(f'Batch {i+1}/{total_batches} processed')
    
#     # Convert accumulated lists to arrays
#     y_pred_accum = np.array(y_pred_accum)
#     y_true_accum = np.array(y_true_accum)
    
#     # Compute confusion matrix
#     conf_matrix = confusion_matrix(y_true_accum, y_pred_accum)
#     return conf_matrix


#############################################################################
# Bigger resenet
#############################################################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.layers import Add, ReLU, AveragePooling2D, Flatten, Dense


def ConvBlock(inputs, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def IdentityBlock(inputs, filters, kernel_size, padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return Add()([x, inputs])


def ResNet18(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
  
    # BLOCK-1
    x = ConvBlock(input_layer, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # BLOCK-2
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op2_1 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    x = ConvBlock(op2_1, filters=64, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op2 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    # BLOCK-3
    x = ConvBlock(op2, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    adjust_op2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op2)
    x = Dropout(0.5)(x)
    op3_1 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    x = ConvBlock(op3_1, filters=128, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op3 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    # BLOCK-4
    x = ConvBlock(op3, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    adjust_op3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op3)
    x = Dropout(0.5)(x)
    op4_1 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    x = ConvBlock(op4_1, filters=256, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op4 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    # BLOCK-5
    x = ConvBlock(op4, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    adjust_op4 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op4)
    x = Dropout(0.5)(x)
    op5_1 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    x = ConvBlock(op5_1, filters=512, kernel_size=(3, 3), padding='same')
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    x = Dropout(0.5)(x)
    op5 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    # FINAL BLOCK
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)



    return model
