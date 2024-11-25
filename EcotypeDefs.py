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
import scipy.signal
import google.cloud.storage
from io import BytesIO
import gcsfs
from tensorflow.keras.callbacks import TensorBoard


#%% Testing generators for gs cloud

# import tensorflow as tf
# import h5py
# import gcsfs

# # Define GCS file path
# hdf5_file_path_val = 'gs://dclde2026_hdf5s/hdf5s/MalahatBalanced_melSpec_8khz_PCEN.h5'

# def data_generator(file_path, group_name='train', batch_size=32):
#     def generator():
#         fs = gcsfs.GCSFileSystem()
#         with fs.open(file_path, 'rb') as f:
#             with h5py.File(f, 'r') as hf:
#                 group = hf[group_name]
#                 keys = list(group.keys())
#                 for i in range(0, len(keys), batch_size):
#                     batch_keys = keys[i:i + batch_size]
#                     spectrograms = [group[key]['spectrogram'][()] for key in batch_keys]
#                     labels = [group[key]['label'][()] for key in batch_keys]
#                     yield tf.convert_to_tensor(spectrograms, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)
#     return generator

# # Create a tf.data.Dataset
# batch_size = 32
# train_dataset = tf.data.Dataset.from_generator(
#     Eco.data_generator(hdf5_file_path_val, group_name='train', batch_size=batch_size),
#     output_signature=(
#         tf.TensorSpec(shape=(batch_size, None, None), dtype=tf.float32),  # Adjust based on actual shape
#         tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
#     )
# )


#%%










#PCEN
def PCEN(spectrogram, alpha =0.5, delta=2, r=0.5, eps =1e-6):
    spectrogram= librosa.db_to_power(spectrogram)

    # # Compute PCEN parameters
    # alpha = 0.5
    # delta = 2
    # r = .5
    # eps = 1e-6
    
    # Compute the power spectrogram smoothed by a short-term averaging filter
    smooth_power_spectrogram = scipy.signal.lfilter([alpha],
                                                    [1, -alpha], 
                                                    spectrogram, axis=1)
    # Compute the gain control filter
    gain_control = np.maximum(eps, smooth_power_spectrogram / 
                              (smooth_power_spectrogram + delta * 
                               np.mean(smooth_power_spectrogram,
                                       axis=1, keepdims=True)))
    
    # Apply the gain control filter to the original power spectrogram
    pcen = (spectrogram ** r) * gain_control - eps
    spectrogram = librosa.power_to_db(pcen, ref=np.max)
    return(spectrogram)



def create_spectrogram(audio, return_snr=False,**kwargs):
    """
    Create the audio representation.

    kwargs options:
        clipDur (float): clip duration (sec).
        nfft (int): FFT window size (samples).
        hop_length (int): Hop length (samples).
        outSR (int): Target sampling rate.
        spec_type (str): Spectrogram type, 'normal' or 'mel'.
        spec_power(int): Magnitude spectrum type, approximate GPL, default =2
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
        'hop_length': 25,     # default hop length (samples)
        'outSR': 16000,         # default target sampling rate
        'spec_type': 'normal',  # default spectrogram type, 'normal' or 'mel'
        'min_freq': None,       # default minimum frequency to retain
        'rowNorm': False,        # normalize the spectrogram rows
        'colNorm': False,        # normalize the spectrogram columns
        'rmDCoffset': True,     # remove DC offset by subtracting mean
        'inSR': None,            # original sample rate of the audio file
        'spec_power':2,
        'returnDB':True,         # return spectrogram in linear or convert to db 
        'PCEN':False,             # Per channel energy normalization
        # PCEN parameters
        'PCEN_power':31,
        'time_constant':.8,
        'eps':1e-6,
        'gain':0.08,
        'power':.25,
        'bias':10,
        'fmin':0,
        'fmax':16000
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
        
        # If no power level specified set to 2 
        if 'spec_power' not in params:
            params['spec_power']=2
            
        spectrogram = np.abs(librosa.stft(audio, 
                                          n_fft=params['nfft'], 
                                          hop_length=params['hop_length']))
        
        # Normalize the spectrogram
        if params['rowNorm']==True:
            row_medians = np.median(spectrogram, axis=1, keepdims=True)
            spectrogram = spectrogram - row_medians
            
        if params['colNorm']==True:
            col_medians = np.median(spectrogram, axis=0, keepdims=True)
            spectrogram = spectrogram - col_medians
            
        spectrogram = np.abs(spectrogram)**params['spec_power']
        #spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        if params['returnDB']==True:
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
    elif params['spec_type'] == 'mel':
        
        melspec = librosa.feature.melspectrogram(y=audio, 
                                                     sr=params['outSR'], 
                                                     n_fft=params['nfft'], 
                                                     hop_length=params['hop_length'],
                                                     fmin =params['fmin'],
                                                     power =params['spec_power'])
        

        
        # PCEN
        if params['PCEN']==True:
            spectrogram =librosa.pcen( 
                S = melspec * (2 ** params['PCEN_power']),
                time_constant=params['time_constant'],
                eps=params['eps'],
                gain =params['gain'],
                power=params['power'],
                bias=params['bias'],
                sr=params['outSR'],
                hop_length=params['hop_length'])
        else:
            spectrogram =melspec
            

        
        # Normalize the spectrogram
        if params['rowNorm']:
            row_medians = np.median(spectrogram, axis=1, keepdims=True)
            spectrogram = spectrogram - row_medians
            
        if params['colNorm']:
            col_medians = np.median(spectrogram, axis=0, keepdims=True)
            spectrogram = spectrogram - col_medians        
        
        if params['returnDB']==True:
            #spectrogram = np.round(spectrogram,2)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    else:
        raise ValueError("Invalid spectrogram type. Supported types are 'normal' and 'mel'.")
    

    # # Trim frequencies if min_freq is specified
    # if params['min_freq'] is not None:
    #     mel_frequencies = librosa.core.mel_frequencies(n_mels=spectrogram.shape[0] + 2)
    #     min_idx = np.argmax(mel_frequencies >= params['min_freq'])
    #     spectrogram = spectrogram[min_idx:, :]
        
    
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
        'outSR': 16000,         # default target sampling rate
    }

    # Update parameters based on kwargs
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        #else:
            #raise TypeError(f"Invalid keyword argument '{key}'")

    # Get the duration of the audio file
    file_duration = librosa.get_duration(path=file_path)
    
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
                                           duration=params['clipDur'],
                                           mono=False)
    # Determine the number of channels
    num_channels = audio_data.shape[0] if audio_data.ndim > 1 else 1
    
    # Retain only the first channel if there are multiple
    if num_channels > 1:
        print(f"Audio has {num_channels} channels. Retaining only the first channel.")
        audio_data = audio_data[0]

    
    # Create audio representation
    spec, snr = create_spectrogram(audio_data, return_snr=return_snr, **kwargs)

    if return_snr:
        return spec, snr
    else:
        return spec




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
          # Store the parameters as attributes
    # Open HDF5 file in write mode
    with h5py.File(hdf5_filename, 'w') as hf:
        # Store parameters as attributes
        for key, value in parms.items():
            if value is not None:
                hf.attrs[key] = value
        
        # Create groups for train and test datasets
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        # Iterate through annotations and create datasets
        for idx, row in annotations.iloc[0:].iterrows():
       # for idx, row in annotations.iterrows():
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

            # Load and process audio segment
            spec, SNR = load_and_process_audio_segment(file_path, start_time, 
                                                       end_time, return_snr=True, **parms)

            # Determine which group to store in (train or test)
            dataset = train_group if traintest == 'Train' else test_group

            # Create datasets for each attribute
            data_group = dataset.create_group(f'{idx}')
            data_group.create_dataset('spectrogram', data=spec)
            data_group.create_dataset('label', data=label)
            data_group.create_dataset('deployment', data=dep)
            data_group.create_dataset('provider', data=provider)
            data_group.create_dataset('KW', data=kw)
            data_group.create_dataset('KW_certain', data=kwCertin)
            data_group.create_dataset('SNR', data=SNR)
            data_group.create_dataset('UTC', data=utc)

            print(f"Processed {idx + 1} of {len(annotations)}")




def create_clip_folder(annotations, fileOutLoc, parms):
    """
    Create folders wtih clips of each label
    """
    # Iterate through annotations and create datasets
    #for idx, row in annotations.iloc[80816:].iterrows():
    for idx, row in annotations.iterrows():
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

        # Load and process audio segment
        spec, SNR = load_and_process_audio_segment(file_path, start_time, 
                                                   end_time, return_snr=True,
                                                   **parms)    




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
                 minFreq = None):
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
        
        # This really shouldn't be used anymore because the data representaion
        # has the meta and as of now, if the frequencies are restricted there is
        # no place to store that info.
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
            return np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels),  num_classes=self.n_classes)
    def __shuffle__(self):
        np.random.shuffle(self.indexes)
        print('shuffled!')
    
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('Epoc end all shuffled!')


# Batch loader for google streaming
class BatchLoaderGScloud(keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size=250, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False,
                 minFreq=None, bucket_name='dclde_2026hdf5s'):
        self.fs = gcsfs.GCSFileSystem()
        self.hf = h5py.File(self.fs.open(hdf5_file, 'rb'), 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels = return_data_labels
        self.data_keys = list(self.hf[trainTest].keys())
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
        self.bucket_name = bucket_name

        # Get the spectrogram size
        self.train_group = self.hf[trainTest]
        self.first_key = list(self.train_group.keys())[0]
        self.data = self.train_group[self.first_key]['spectrogram']
        self.specSize = self.data.shape

        # Frequency limit logic
        self.minFreq = minFreq
        self.minIdx = 0
        if math.isfinite(self.minFreq):
            mel_frequencies = librosa.core.mel_frequencies(n_mels=self.specSize[0] + 2)
            self.minIdx = np.argmax(mel_frequencies >= self.minFreq)
            self.data = self.data[self.minIdx:, :]
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
            spec = self.hf[self.trainTest][key]['spectrogram'][self.minIdx:, :]
            label = self.hf[self.trainTest][key]['label'][()]
            batch_data.append(spec)
            batch_labels.append(label)

        if self.return_data_labels:
            return batch_data, batch_labels
        else:
            return np.array(batch_data), keras.utils.to_categorical(np.array(batch_labels), num_classes=self.n_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('Epoch end - all shuffled!')

    def close(self):
        self.hf.close()
        print("HDF5 file closed.")



# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical


# class BatchLoader3:
#     def __init__(self, hdf5_file, batch_size=250, trainTest='train',
#                  shuffle=True, n_classes=7, return_data_labels=False,
#                  minFreq='nan'):
#         self.hf = h5py.File(hdf5_file, 'r')
#         self.batch_size = batch_size
#         self.trainTest = trainTest
#         self.shuffle = shuffle
#         self.n_classes = n_classes
#         self.return_data_labels = return_data_labels
#         self.data_keys = list(self.hf[trainTest].keys())
#         self.num_samples = len(self.data_keys)
        
#         # Get the spectrogram size from the first sample
#         first_key = self.data_keys[0]
#         self.data = self.hf[trainTest][first_key]['spectrogram']
#         self.specSize = self.data.shape
        
#         # Handle minimum frequency
#         self.minFreq = minFreq
#         if math.isfinite(self.minFreq):
#             mel_frequencies = librosa.core.mel_frequencies(n_mels=self.specSize[0] + 2)
#             self.minIdx = np.argmax(mel_frequencies >= self.minFreq)
#             self.specSize = self.data[self.minIdx:, :].shape
        
#         if self.shuffle:
#             np.random.shuffle(self.data_keys)

#     def __len__(self):
#         return int(np.ceil(self.num_samples / self.batch_size))

#     def generate_dataset(self):
#         # Create a list of (data, label) tuples
#         dataset = []
#         for key in self.data_keys:
#             spec = self.hf[self.trainTest][key]['spectrogram'][self.minIdx:, :]
#             label = self.hf[self.trainTest][key]['label'][()]
#             dataset.append((spec, label))

#         # Shuffle dataset if needed
#         if self.shuffle:
#             np.random.shuffle(dataset)

#         # Separate data and labels
#         data_list, labels_list = zip(*dataset)

#         # Convert lists to numpy arrays
#         data_array = np.array(data_list)
#         labels_array = np.array(labels_list)

#         # If returning data and labels, return as tuple
#         if self.return_data_labels:
#             return data_array, labels_array

#         # Otherwise, return as TensorFlow Dataset
#         return tf.data.Dataset.from_tensor_slices((data_array, to_categorical(labels_array, num_classes=self.n_classes)))

# from tensorflow.keras import backend as K
# def custom_weighted_loss(class_weights, priorities):
#     """
#     Custom weighted categorical crossentropy loss function.

#     Parameters:
#     - class_weights: Dictionary containing weights for each class.
#     - priorities: List of lists where each sublist represents a group of classes with a priority.

#     Returns:
#     - loss function to be used in model compilation.
#     """
#     def loss(y_true, y_pred):
#         # Convert class weights to tensor
#         weights = tf.constant([class_weights[key] for key in sorted(class_weights.keys())], dtype=tf.float32)
        
#         # Convert priorities to a tensor
#         priority_indices = [sorted(class_weights.keys()).index(cls) for priority_group in priorities for cls in priority_group]
#         priority_mask = K.one_hot(priority_indices, len(class_weights))  # Remove dtype argument
        
#         # Calculate cross entropy
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clip to prevent log(0) instability
#         cross_entropy = -y_true * K.log(y_pred)
        
#         # Apply weights and priorities
#         weighted_cross_entropy = cross_entropy * weights
#         weighted_cross_entropy *= priority_mask
        
#         # Sum over classes and mean over batch
#         return K.mean(K.sum(weighted_cross_entropy, axis=-1))
    
#     return loss

def train_model(model, train_generator, val_generator, epochs, 
                tensorBoard=False):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3, 
                                   restore_best_weights=True)
    
    if tensorBoard:

        # Define the TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        

        # Train the model with early stopping
        model.fit(x=train_generator,  # Here, `train_generator` should be a generator object
                  epochs=epochs,
                  validation_data=val_generator,
                  callbacks=[early_stopping, tensorboard_callback])
    else:
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


from sklearn.metrics import confusion_matrix, accuracy_score

# So this is well and good but I'd like to create be able to look at the timestamps
# from the predictions so we can do something lie softmax


def batch_conf_matrix(loaded_model, val_batch_loader, label_dict):
    """
    Computes a confusion matrix with human-readable labels, percentages
    formatted to two decimal places, and overall accuracy.
    
    When you forget how to create the label dictionary here is an example
    label_dict = dict(zip(AllAnno['label'], AllAnno['Labels']))

    Args:
        loaded_model: The trained model to evaluate.
        val_batch_loader: The batch loader providing validation data and labels.
        label_dict: Dictionary mapping numeric labels to human-readable labels.

    Returns:
        conf_matrix_str: Confusion matrix with human-readable labels as a DataFrame.
        accuracy: Overall accuracy in percentage.
        results_df: A DataFrame containing true labels and predicted labels.
    """
    y_pred_accum = []
    y_true_accum = []

    total_batches = len(val_batch_loader)

    for i in range(total_batches):
        batch_data = val_batch_loader.__getitem__(i)
        batch_pred = loaded_model.predict(batch_data[0])
        batch_pred_labels = np.argmax(batch_pred, axis=1)
        batch_true_labels = np.array(batch_data[1])
        batch_true_labels = np.argmax(batch_true_labels, axis=1)

        y_pred_accum.extend(batch_pred_labels)
        y_true_accum.extend(batch_true_labels)

        print(f'Batch {i+1}/{total_batches} processed')

    y_pred_accum = np.array(y_pred_accum)
    y_true_accum = np.array(y_true_accum)

    # Compute raw confusion matrix
    conf_matrix_raw = confusion_matrix(y_true_accum, y_pred_accum)

    # Normalize confusion matrix by rows to get percentages
    conf_matrix_percent = conf_matrix_raw.astype(np.float64)  # Convert to float for division
    row_sums = conf_matrix_raw.sum(axis=1, keepdims=True)     # Row-wise sums
    conf_matrix_percent = np.divide(conf_matrix_percent, row_sums, where=row_sums != 0) * 100  # Avoid division by zero

    # Map numeric labels to human-readable labels
    unique_labels = sorted(set(y_true_accum) | set(y_pred_accum))  # All unique labels in validation data
    human_labels = [label_dict[label] for label in unique_labels]

    # Format confusion matrix to two decimal places
    conf_matrix_percent_formatted = np.array([[f"{value:.2f}" for value in row] for row in conf_matrix_percent])

    # Create DataFrame with human-readable labels
    conf_matrix_df = pd.DataFrame(conf_matrix_percent_formatted, index=human_labels, columns=human_labels)

    # Compute overall accuracy
    accuracy = accuracy_score(y_true_accum, y_pred_accum) 

    # Create a DataFrame for analysis
    results_df = pd.DataFrame({
        'True Label': [label_dict[label] for label in y_true_accum],
        'Predicted Label': [label_dict[label] for label in y_pred_accum]
    })

    print("Confusion Matrix (Percentages, Two Decimals):")
    print(conf_matrix_df)
    print(f"Overall Accuracy: {accuracy:.2f}%")

    return conf_matrix_df, conf_matrix_raw, accuracy, results_df


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_score_distributions(loaded_model, val_batch_loader, label_dict):
    """
    Generates paired violin plots of score distributions for true positives (TP) and false positives (FP)
    for each class in the validation dataset.

    Args:
        loaded_model: The trained model to evaluate.
        val_batch_loader: The batch loader providing validation data and labels.
        label_dict: Dictionary mapping numeric labels to human-readable labels.
    
    Returns:
        score_df: A DataFrame with scores and classification information.
    """
    # Initialize accumulators
    y_true_accum = []
    y_pred_accum = []
    score_accum = []

    total_batches = len(val_batch_loader)
    # Iterate through the batches in the validation loader
    for i in range(len(val_batch_loader)):
        batch_data = val_batch_loader.__getitem__(i)
        batch_scores = loaded_model.predict(batch_data[0])  # Raw model outputs (softmax scores)
        batch_pred_labels = np.argmax(batch_scores, axis=1)
        batch_true_labels = np.argmax(batch_data[1], axis=1)

        # Accumulate true labels, predicted labels, and scores
        y_true_accum.extend(batch_true_labels)
        y_pred_accum.extend(batch_pred_labels)
        score_accum.extend(batch_scores)
        print(f'Batch {i+1}/{total_batches} processed')

    # Convert accumulated lists to arrays
    y_true_accum = np.array(y_true_accum)
    y_pred_accum = np.array(y_pred_accum)
    score_accum = np.array(score_accum)

    # Prepare the DataFrame
    score_data = []
    for i, true_label in enumerate(y_true_accum):
        pred_label = y_pred_accum[i]
        scores = score_accum[i]

        for class_label, score in enumerate(scores):
            label_type = "True Positive" if (true_label == class_label == pred_label) else "False Positive"
            score_data.append({
                "Class": label_dict[class_label],
                "Score": score,
                "Type": label_type
            })

    score_df = pd.DataFrame(score_data)

    # Plot the paired violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=score_df, x="Class", y="Score", hue="Type", split=True, inner="quartile", palette="muted")
    plt.title("Score Distributions for True Positives and False Positives")
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.legend(title="Type")
    plt.tight_layout()
    plt.show()

    return score_df









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
#########################################################################
#Pipeline for producing streaming detections
##########################################################################

import soundfile as sf

import sounddevice as sd  # For capturing real-time audio


import os
import soundfile as sf
import numpy as np
from EcotypeDefs import create_spectrogram  # Assuming this function is defined in EcotypeDefs.py
import sounddevice as sd  # For capturing real-time audio

from keras.models import load_model
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

class AudioProcessor:
    def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
                  params=None, model=None, detection_thresholds=None,
                  selection_table_name="detections.txt", class_names=None):
        self.folder_path = folder_path
        self.audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))] if folder_path else []
        self.segment_duration = segment_duration  # Duration of each segment in seconds
        self.overlap = overlap  # Overlap between segments in seconds
        self.params = params if params is not None else {
            'outSR': 16000,
            'clipDur': segment_duration,
            'nfft': 512,
            'hop_length': 3200,
            'spec_type': 'mel',  # Assuming mel spectrogram is used
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None
        }
        self.model = model
        self.model_input_shape = model.input_shape[1:] if model else None
        
        # Initialize variables for real-time streaming
        self.buffer = np.array([], dtype='float32')  # Buffer to accumulate audio chunks
        self.sr = None  # Sample rate of the incoming audio stream
        
        # Detection thresholds for each class
        self.detection_thresholds = detection_thresholds if detection_thresholds else {
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 0.5,
            6: 0.5,
        }
        
        # Dictionary to map class IDs to names
        self.class_names = class_names if class_names else {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio',
        }
        
        # Dictionary to track ongoing detections
        self.ongoing_detections = {class_id: None for class_id in range(len(self.detection_thresholds))}
        
        # Selection table file name
        self.selection_table_name = selection_table_name
        
        # Initialize or create the selection table file
        self.init_selection_table()

        # Counter for tracking the number of detections
        self.detection_counter = 0
        
        # Initialize segment_start_time for the entire stream
        self.segment_start_time = 0.0

    def init_selection_table(self):
        # Create or overwrite the selection table file with headers
        with open(self.selection_table_name, 'w') as f:
            f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\n")
    
    def load_audio_generator(self, filename):
        audio_path = os.path.join(self.folder_path, filename)
        y, sr = sf.read(audio_path, dtype='float32')
        self.params['inSR'] = sr  # Update the input sample rate
        return y, sr

    def create_segments(self, y, sr):
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        file_duration = len(y) / sr  # Total duration of the audio file in seconds
    
        # Initialize start time relative to the file
        start_time = 0.0
    
        for start in range(0, len(y) - segment_length + 1, segment_length - overlap_length):
            yield y[start:start + segment_length], start_time  # Yield segment and start time in seconds relative to the file
            start_time += (segment_length - overlap_length) / sr  # Update start time for the next segment
            
    def create_spectrogram(self, y, sr):
        if 'inSR' not in self.params or self.params['inSR'] is None:
            raise ValueError("Input sample rate 'inSR' must be set before creating spectrograms.")
        expected_time_steps = self.model_input_shape[1] if self.model_input_shape else None
        
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        
        if expected_time_steps and spectrogram.shape[1] < expected_time_steps:
            spectrogram = np.pad(spectrogram, ((0, 0), 
                                                (0, expected_time_steps - 
                                                spectrogram.shape[1])), 
                                  mode='constant')
        
        elif spectrogram.shape[1] > expected_time_steps:
            spectrogram = spectrogram[:, :expected_time_steps]
        
        return spectrogram

    def process_audio_chunk(self, chunk):
        self.buffer = np.append(self.buffer, chunk)
        if self.sr is None:
            self.sr = sd.query_devices(None, 'input')['default_samplerate']
    
        segment_length = int(self.segment_duration * self.sr)
        overlap_length = int(self.overlap * self.sr)
    
        while len(self.buffer) >= segment_length:
            segment = self.buffer[:segment_length]
            self.buffer = self.buffer[segment_length - overlap_length:]
    
            spectrogram = self.create_spectrogram(segment, self.sr)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            predictions = self.model.predict(spectrogram)[0]  # Assuming batch size 1
    
            # Process predictions for each class
            for class_id, prediction_score in enumerate(predictions):
                detection_threshold = self.detection_thresholds[class_id]
    
                if prediction_score >= detection_threshold:
                    # Start or merge ongoing detection
                    if self.ongoing_detections[class_id] is None:
                        # Start new detection
                        self.ongoing_detections[class_id] = {
                            'start_time': self.segment_start_time,
                            'end_time': self.segment_start_time + self.segment_duration,
                            'class_id': class_id,
                        }
                    else:
                        # Merge with ongoing detection
                        self.ongoing_detections[class_id]['end_time'] = self.segment_start_time + self.segment_duration
                else:
                    # End ongoing detection
                    if self.ongoing_detections[class_id] is not None:
                        # Output the detection to selection table
                        self.output_detection(self.ongoing_detections[class_id])
                        self.ongoing_detections[class_id] = None
    
            # Increment segment start time by subtracting overlap
            self.segment_start_time += self.segment_duration - self.overlap
    
    
    def output_detection(self, detection):
        self.detection_counter += 1
        
        with open(self.selection_table_name, 'a') as f:
            selection = self.detection_counter
            start_time = detection['start_time']
            end_time = detection['end_time']
            class_id = detection['class_id']
            class_name = self.class_names[class_id]
            
            f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\n")
            

    def process_all_files(self):
        for filename in self.audio_files:
            print(f"Processing file: {filename}")
            
            # Load audio file and process segments
            audio_generator, sr = self.load_audio_generator(filename)
            segments = self.create_segments(audio_generator, sr)
            
            # Process each segment
            for segment, _ in segments:
                self.process_audio_chunk(segment)


import os
import numpy as np
import soundfile as sf
import sounddevice as sd

class AudioProcessor2:
    def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
                 params=None, model=None, detection_thresholds=None,
                 selection_table_name="detections.txt", class_names=None,
                 table_type="selection"):  # New parameter 'table_type'
        self.table_type = table_type
        self.folder_path = folder_path
        self.audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))] if folder_path else []
        self.segment_duration = segment_duration  # Duration of each segment in seconds
        self.overlap = overlap  # Overlap between segments in seconds
        self.params = params if params is not None else {
            'outSR': 16000,
            'clipDur': segment_duration,
            'nfft': 512,
            'hop_length': 3200,
            'spec_type': 'mel',  # Assuming mel spectrogram is used
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None
        }
        self.model = model
        self.model_input_shape = model.input_shape[1:] if model else None
        
        # Initialize variables for real-time streaming
        self.buffer = np.array([], dtype='float32')  # Buffer to accumulate audio chunks
        self.sr = None  # Sample rate of the incoming audio stream
        
        # Detection thresholds for each class
        self.detection_thresholds = detection_thresholds if detection_thresholds else {
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 0.5,
            6: 0.5,
        }
        
        # Dictionary to map class IDs to names
        self.class_names = class_names if class_names else {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio',
        }
        
        # Dictionary to track ongoing detections
        self.ongoing_detections = {class_id: None for class_id in range(len(self.detection_thresholds))}
        
        # Selection table file name
        self.selection_table_name = selection_table_name
        
        # Initialize or create the selection table file
        self.init_selection_table()

        # Counter for tracking the number of detections
        self.detection_counter = 0
        
        # Initialize segment_start_time for the entire stream
        self.segment_start_time = 0.0
        

    def init_selection_table(self):
        # Create or overwrite the selection table file with headers
        with open(self.selection_table_name, 'w') as f:
            if self.table_type == "selection":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\n")
            elif self.table_type == "sound":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\tSound\n")
    
    def load_audio_generator(self, filename):
        audio_path = os.path.join(self.folder_path, filename)
        y, sr = sf.read(audio_path, dtype='float32')
        self.params['inSR'] = sr  # Update the input sample rate
        return y, sr

    def create_segments(self, y, sr):
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        file_duration = len(y) / sr  # Total duration of the audio file in seconds
    
        # Initialize start time relative to the file
        start_time = 0.0
    
        for start in range(0, len(y) - segment_length + 1, segment_length - overlap_length):
            yield y[start:start + segment_length], start_time  # Yield segment and start time in seconds relative to the file
            start_time += (segment_length - overlap_length) / sr  # Update start time for the next segment
            
    def create_spectrogram(self, y, sr):
        if 'inSR' not in self.params or self.params['inSR'] is None:
            raise ValueError("Input sample rate 'inSR' must be set before creating spectrograms.")
        expected_time_steps = self.model_input_shape[1] if self.model_input_shape else None
        
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        
        if expected_time_steps and spectrogram.shape[1] < expected_time_steps:
            spectrogram = np.pad(spectrogram, ((0, 0), 
                                                (0, expected_time_steps - 
                                                spectrogram.shape[1])), 
                                  mode='constant')
        
        elif spectrogram.shape[1] > expected_time_steps:
            spectrogram = spectrogram[:, :expected_time_steps]
        
        return spectrogram

    def process_audio_chunk(self, chunk, audio_filename):
        self.buffer = np.append(self.buffer, chunk)
        if self.sr is None:
            self.sr = sd.query_devices(None, 'input')['default_samplerate']

        segment_length = int(self.segment_duration * self.sr)
        overlap_length = int(self.overlap * self.sr)

        while len(self.buffer) >= segment_length:
            segment = self.buffer[:segment_length]
            self.buffer = self.buffer[segment_length - overlap_length:]

            spectrogram = self.create_spectrogram(segment, self.sr)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            predictions = self.model.predict(spectrogram)[0]  # Assuming batch size 1

            for class_id, prediction_score in enumerate(predictions):
                detection_threshold = self.detection_thresholds[class_id]

                if prediction_score >= detection_threshold:
                    if self.ongoing_detections[class_id] is None:
                        self.ongoing_detections[class_id] = {
                            'start_time': self.segment_start_time,
                            'end_time': self.segment_start_time + self.segment_duration,
                            'class_id': class_id,
                            'score': prediction_score  # Store the prediction score
                        }
                    else:
                        self.ongoing_detections[class_id]['end_time'] = self.segment_start_time + self.segment_duration

                else:
                    if self.ongoing_detections[class_id] is not None:
                        self.output_detection(self.ongoing_detections[class_id], audio_filename)
                        self.ongoing_detections[class_id] = None

            self.segment_start_time += self.segment_duration - self.overlap
    
    
    def output_detection(self, detection, audio_filename):
        self.detection_counter += 1
    
        with open(self.selection_table_name, 'a') as f:
            selection = self.detection_counter
            start_time = detection['start_time']
            end_time = detection['end_time']
            class_id = detection['class_id']
            class_name = self.class_names[class_id]
            score = np.round(np.mean(detection['score']),3)
    
            if self.table_type == "selection":
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\n")
            elif self.table_type == "sound":
                # Write the audio file name instead of audio samples
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\t{audio_filename}\n")
              

    def process_all_files(self):
        for filename in self.audio_files:
            print(f"Processing file: {filename}")
            
            # Load audio file and process segments
            audio_generator, sr = self.load_audio_generator(filename)
            segments = self.create_segments(audio_generator, sr)
            
            # Process each segment
            for segment, _ in segments:
                self.process_audio_chunk(segment, filename)



class AudioProcessor3:
    def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
                 params=None, model=None, detection_thresholds=None,
                 selection_table_name="detections.txt", class_names=None,
                 table_type="selection"):  # New parameter 'table_type'
        self.table_type = table_type
        self.folder_path = folder_path
        self.audio_files = self.find_audio_files(folder_path) if folder_path else []
        self.segment_duration = segment_duration  # Duration of each segment in seconds
        self.overlap = overlap  # Overlap between segments in seconds
        self.params = params if params is not None else {
            'outSR': 16000,
            'clipDur': segment_duration,
            'nfft': 512,
            'hop_length': 3200,
            'spec_type': 'mel',  # Assuming mel spectrogram is used
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None
        }
        self.model = model
        self.model_input_shape = model.input_shape[1:] if model else None
        
        # Initialize variables for real-time streaming
        self.buffer = np.array([], dtype='float32')  # Buffer to accumulate audio chunks
        self.sr = None  # Sample rate of the incoming audio stream
        
        # Detection thresholds for each class
        self.detection_thresholds = detection_thresholds if detection_thresholds else {
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 0.5,
            6: 0.5,
        }
        
        # Dictionary to map class IDs to names
        self.class_names = class_names if class_names else {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio',
        }
        
        # Dictionary to track ongoing detections
        self.ongoing_detections = {class_id: None for class_id in range(len(self.detection_thresholds))}
        
        # Selection table file name
        self.selection_table_name = selection_table_name
        
        # Initialize or create the selection table file
        self.init_selection_table()

        # Counter for tracking the number of detections
        self.detection_counter = 0
        
        # Initialize segment_start_time for the entire stream
        self.segment_start_time = 0.0

    def find_audio_files(self, folder_path):
        audio_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def init_selection_table(self):
        # Create or overwrite the selection table file with headers
        with open(self.selection_table_name, 'w') as f:
            if self.table_type == "selection":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\n")
            elif self.table_type == "sound":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\tSound\n")
    
    def load_audio_generator(self, filename):
        y, sr = sf.read(filename, dtype='float32')
        self.params['inSR'] = sr  # Update the input sample rate
        return y, sr

    def create_segments(self, y, sr):
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        file_duration = len(y) / sr  # Total duration of the audio file in seconds
    
        # Initialize start time relative to the file
        start_time = 0.0
    
        for start in range(0, len(y) - segment_length + 1, segment_length - overlap_length):
            yield y[start:start + segment_length], start_time  # Yield segment and start time in seconds relative to the file
            start_time += (segment_length - overlap_length) / sr  # Update start time for the next segment
            
    def create_spectrogram(self, y, sr):
        if 'inSR' not in self.params or self.params['inSR'] is None:
            raise ValueError("Input sample rate 'inSR' must be set before creating spectrograms.")
        expected_time_steps = self.model_input_shape[1] if self.model_input_shape else None
        
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        
        if expected_time_steps and spectrogram.shape[1] < expected_time_steps:
            spectrogram = np.pad(spectrogram, ((0, 0), 
                                                (0, expected_time_steps - 
                                                spectrogram.shape[1])), 
                                  mode='constant')
        
        elif spectrogram.shape[1] > expected_time_steps:
            spectrogram = spectrogram[:, :expected_time_steps]
        
        return spectrogram

    def process_audio_chunk(self, chunk, audio_filename):
        self.buffer = np.append(self.buffer, chunk)
        if self.sr is None:
            self.sr = sd.query_devices(None, 'input')['default_samplerate']

        segment_length = int(self.segment_duration * self.sr)
        overlap_length = int(self.overlap * self.sr)

        while len(self.buffer) >= segment_length:
            segment = self.buffer[:segment_length]
            self.buffer = self.buffer[segment_length - overlap_length:]

            spectrogram = self.create_spectrogram(segment, self.sr)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            predictions = self.model.predict(spectrogram)[0]  # Assuming batch size 1

            for class_id, prediction_score in enumerate(predictions):
                detection_threshold = self.detection_thresholds[class_id]

                if prediction_score >= detection_threshold:
                    if self.ongoing_detections[class_id] is None:
                        self.ongoing_detections[class_id] = {
                            'start_time': self.segment_start_time,
                            'end_time': self.segment_start_time + self.segment_duration,
                            'class_id': class_id,
                            'score': prediction_score  # Store the prediction score
                        }
                    else:
                        self.ongoing_detections[class_id]['end_time'] = self.segment_start_time + self.segment_duration

                else:
                    if self.ongoing_detections[class_id] is not None:
                        self.output_detection(self.ongoing_detections[class_id], audio_filename)
                        self.ongoing_detections[class_id] = None

            self.segment_start_time += self.segment_duration - self.overlap
    
    
    def output_detection(self, detection, audio_filename):
        self.detection_counter += 1
    
        with open(self.selection_table_name, 'a') as f:
            selection = self.detection_counter
            start_time = detection['start_time']
            end_time = detection['end_time']
            class_id = detection['class_id']
            class_name = self.class_names[class_id]
            score = np.round(np.mean(detection['score']),3)
    
            if self.table_type == "selection":
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\n")
            elif self.table_type == "sound":
                # Write the audio file name instead of audio samples
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\t{audio_filename}\n")
              

    def process_all_files(self):
        for filename in self.audio_files:
            print(f"Processing file: {filename}")
            
            # Load audio file and process segments
            audio_generator, sr = self.load_audio_generator(filename)
            segments = self.create_segments(audio_generator, sr)
            
            # Process each segment
            for segment, _ in segments:
                self.process_audio_chunk(segment, filename)



if __name__ == "__main__":
    
    # Example how to call
    
    # Load Keras model
    folder_path = 'E:\\Malahat\\STN3\\20151028'
    
    # Load Keras model
    model_path = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Models\\20200818\\output_resnet18_PECN_melSpec.keras'
    model = load_model(model_path)

    # Spectrogram parameters
    AudioParms = {
        'clipDur': 2,
        'outSR': 16000,
        'nfft': 512,
        'hop_length': 25,
        'spec_type': 'mel',  
        'spec_power': 2,
        'rowNorm': False,
        'colNorm': False,
        'rmDCoffset': False,
        'inSR': None, 
        'PCEN': True,
        'fmin': 150
    }

    # Example detection thresholds (adjust as needed)
    detection_thresholds = {
        0: 0.8,  # Example threshold for class 0
        1: 0.8,  # Example threshold for class 1
        2: 0.9,  # Example threshold for class 2
        3: 0.8,  # Example threshold for class 3
        4: 0.8,  # Example threshold for class 4
        5: 0.8,  # Example threshold for class 5
        6: 0.9   # Example threshold for class 6
    }

    class_names = {
        0: 'Abiotic',
        1: 'BKW',
        2: 'HW',
        3: 'NRKW',
        4: 'Offshore',
        5: 'SRKW',
        6: 'Und Bio'
    }

    
    
    # Initialize the AudioProcessor with your model and detection thresholds
    processor = AudioProcessor2(folder_path=folder_path, model=model,
                               detection_thresholds=detection_thresholds, 
                               class_names=class_names,
                               params= AudioParms,
                               table_type="sound",
                               overlap=0.25)
    
    # Process all audio files in the directory
    processor.process_all_files()




