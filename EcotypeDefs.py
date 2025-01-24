# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:58:40 2024

Script containing functions and methods for building H5 databases,
building model arechetectur, training and evaluaiton of models


@author: kaity
"""


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
import keras
from keras import Model
from keras.callbacks import EarlyStopping
import scipy.signal
#import gcsfs
from tensorflow.keras.callbacks import TensorBoard


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
        'PCEN_power':31,
        'time_constant':.8,
        'eps':1e-6,
        'gain':0.08,
        'power':.25,
        'bias':10,
        'fmin':0,
        'fmax':16000,
        'Scale Spectrogram': False, # scale the spectrogram between 0 and 1
        'AnnotationTrain': 'bla',
        'AnnotationsTest': 'bla',
        'AnnotationsVal': 'bla',
        'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
        'using batch norm and Raven parameters with Mel Spectrograms and PCEN '
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
        
    if params['Scale Spectrogram'] == True:
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
        
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
    #duration = end_time - start_time
    
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


#############################################
# Create HDF5 with parallelization
################################################


from concurrent.futures import ProcessPoolExecutor

def process_audio_segment(row, parms):
    """
    Process a single audio segment and return the results.
    
    Parameters:
        row (pd.Series): A row from the annotations DataFrame.
        parms (dict): Parameters for processing the audio.
        
    Returns:
        dict: A dictionary containing the processed results.
    """
    try:
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
        spec, SNR = load_and_process_audio_segment(file_path, start_time, end_time, return_snr=True, **parms)

        return {
            'traintest': traintest,
            'data': {
                'spectrogram': spec,
                'label': label,
                'deployment': dep,
                'provider': provider,
                'KW': kw,
                'KW_certain': kwCertin,
                'SNR': SNR,
                'UTC': utc
            }
        }
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None

def process_audio_segment_wrapper(args):
    """
    Wrapper for process_audio_segment to allow passing arguments for multiprocessing.
    
    Parameters:
        args (tuple): A tuple containing (row, parms).
        
    Returns:
        dict: Processed results from process_audio_segment.
    """
    row, parms = args
    return process_audio_segment(row, parms)

def process_batch(batch, parms):
    """
    Process a batch of annotations in parallel.
    
    Parameters:
        batch (pd.DataFrame): A batch of annotations.
        parms (dict): Parameters for processing the audio.
        
    Returns:
        list: Processed results for the batch.
    """
    with ProcessPoolExecutor() as executor:
        # Pass rows and parameters as tuples to the wrapper function
        results = list(executor.map(process_audio_segment_wrapper, [(row, parms) for _, row in batch.iterrows()]))
    return list(filter(None, results))  # Exclude failed processes (None values)

def create_hdf5_dataset_parallel2(annotations, hdf5_filename, parms, batch_size=100):
    """
    Create an HDF5 database with spectrogram images for DCLDE data using batch processing.
    
    Parameters:
        annotations (pd.DataFrame): Annotations DataFrame.
        hdf5_filename (str): Output HDF5 file name.
        parms (dict): Parameters for processing the audio.
        batch_size (int): Number of rows to process in each batch.
    """
    with h5py.File(hdf5_filename, 'w') as hf:
        # Store parameters as attributes
        for key, value in parms.items():
            if value is not None:
                hf.attrs[key] = value

        # Create groups for train and test datasets
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')

        # Process annotations in batches
        num_batches = int(np.ceil(len(annotations) / batch_size))
        for batch_idx in range(num_batches):
            print(f"Processing batch {batch_idx + 1}/{num_batches}...")
            batch = annotations.iloc[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_results = process_batch(batch, parms)

            # Write batch results to HDF5 file
            for result in batch_results:
                dataset = train_group if result['traintest'] == 'Train' else test_group
                group_idx = len(dataset)  # Determine next available group index
                data_group = dataset.create_group(f"{group_idx}")

                for key, value in result['data'].items():
                    data_group.create_dataset(key, data=value)

            print(f"Completed batch {batch_idx + 1}/{num_batches}")

    print("HDF5 dataset creation completed.")



#####################################

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
        #label = row['label']
        #traintest = row['traintest']
        #dep = row['Dep']
        #provider = row['Provider']
        #kw = row['KW']
        #kwCertin = row['KW_certain']
        #utc = row['UTC']

        # Load and process audio segment
        spec, SNR = load_and_process_audio_segment(file_path, start_time, 
                                                   end_time, return_snr=True,
                                                   **parms)    


import librosa
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
        
        # # This really shouldn't be used anymore because the data representaion
        # # has the meta and as of now, if the frequencies are restricted there is
        # # no place to store that info.
        # #This is for trimming the frequency range
        # if math.isfinite(self.minFreq):
        #      mel_frequencies = librosa.core.mel_frequencies(n_mels= self.specSize[0]+2)

        #      # Find the index corresponding to 500 Hz in mel_frequencies
        #      self.minIdx = np.argmax(mel_frequencies >= self.minFreq)
            
        #     # # # also update the spectrogram size
        #      self.data= self.data[self.minIdx:,:]
        #      self.specSize = self.data.shape
     
        
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


def train_model_history(model, train_generator, val_generator, epochs, tensorBoard=False):
    '''
    Train model function (same as train_model()) but preserves history
    

    Parameters
    ----------
    model : keras model
        trained Keras model.
    train_generator : 
        train batch generator created with BatchLoader2 .
    val_generator : TYPE
        train batch generator created with BatchLoader2 .
    epochs : int
        Number of epocs to run.
    tensorBoard : bool, optional (False)
        Whether to try to run tensorboard. Not presently working on windows 
        machine

    Returns
    -------
    '''
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3, 
                                   restore_best_weights=True)
    
    if tensorBoard:
        # Define the TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        
        # Train the model and capture history
        history = model.fit(x=train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            callbacks=[early_stopping, tensorboard_callback])
    else:
        # Train the model and capture history
        history = model.fit(x=train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            callbacks=[early_stopping])

    return history    



def plot_training_curves(history):
    '''
    Plots training and validation accuracy/loss curves. See also 


    Parameters
    ----------
    history : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


    

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

from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D

def create_resnet50(input_shape, num_classes):
    # Use a custom input tensor
    input_tensor = Input(shape=input_shape)
    # Create the base model with the modified input tensor
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    #base_model.summary()
    
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # Add global average pooling to reduce feature map dimensions
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Add a fully connected layer for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)
    #model.summary()
    return model

def create_resnet101(input_shape, num_classes):
    # Use a custom input tensor
    input_tensor = Input(shape=input_shape)
    # Create the base model with the modified input tensor
    base_model = keras.applications.ResNet101(include_top=False, weights=None, input_tensor=input_tensor)
    #base_model.summary()
    
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # Add global average pooling to reduce feature map dimensions
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Add a fully connected layer for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)
    model.summary()
    return model



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


# Compile model
def compile_model(model, loss_val = 'categorical_crossentropy'):
    model.compile(loss=loss_val,
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


#############################################################################
# Bigger resenet
#############################################################################
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import  ReLU


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



def ResNet1_testing(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
  
    # BLOCK-1
    x = ConvBlock(input_layer, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

   
    # FINAL BLOCK
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)



    return model





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



def ResNet18_batchNorm(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
  
    # BLOCK-1
    x = ConvBlock(input_layer, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # BLOCK-2
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op2_1 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    x = ConvBlock(op2_1, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=64, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op2 = IdentityBlock(x, filters=64, kernel_size=(3, 3))

    # BLOCK-3
    x = ConvBlock(op2, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    adjust_op2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op2)
    x = Dropout(0.5)(x)
    op3_1 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    x = ConvBlock(op3_1, filters=128, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=128, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op3 = IdentityBlock(x, filters=128, kernel_size=(3, 3))

    # BLOCK-4
    x = ConvBlock(op3, filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    adjust_op3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op3)
    x = Dropout(0.5)(x)
    op4_1 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    x = ConvBlock(op4_1, filters=256, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=256, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op4 = IdentityBlock(x, filters=256, kernel_size=(3, 3))

    # BLOCK-5
    x = ConvBlock(op4, filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    adjust_op4 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='valid')(op4)
    x = Dropout(0.5)(x)
    op5_1 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    x = ConvBlock(op5_1, filters=512, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = ConvBlock(x, filters=512, kernel_size=(3, 3), padding='same')
    x = BatchNormalization()(x)  # Add Batch Normalization
    x = Dropout(0.5)(x)
    op5 = IdentityBlock(x, filters=512, kernel_size=(3, 3))

    # FINAL BLOCK
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)  # Optional Batch Normalization for Dense layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

import json    
def saveModelData(model,modelName, savePath, metadata):
    '''
    Function to save the model with the associated metadata
    
    Parameters
    ----------
    model : keras model
    modelName : string
        name of the model excluding keras
    savePath : string
        Full path file for save model location including model name
    metadata : dictionary
        dictionary containing the parameters used for training excluding
        the hdf5 files for training and validation

    Returns
    -------
    None.
    
    Example :
        metadata = {
            "h5TrainTest": "spectrogram_8kHz_norm01_fft256_hop128.h5",
            "h5TrainEval": "spectrogram_8kHz_norm01_fft256_hop128.h5",
            "parameters": {
                "epochs": 20,
                "batch_size": 32,
                "optimizer": "adam"
            }
        }

    '''
    model.save(savePath + modelName + '.keras')
    with open(savePath + modelName + '_metadata.json', 'w') as f:
        json.dump(metadata, f)
        
#########################################################################
# Model Evaluation Class
#########################################################################

#import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# So this is well and good but I'd like to create be able to look at the timestamps
# from the predictions so we can do something lie softmax

def plot_training_curves(history):
    """
    Plots training and validation accuracy/loss curves.

    Parameters:
        history: History object returned by model.fit().
    """
    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score

class ModelEvaluator:
    def __init__(self, loaded_model, val_batch_loader, label_dict):
        """
        Class to evaluate the trained model on the validataion H5 file
        Initialize the ModelEvaluator object.
        
        Parameters
        ----------
            loaded_model : keras model
            The trained keras model to evaluate
            
            val_batch_loader : object
            Batch loader to feed spectrograms from the H5 model to the kerask 
            model for prediction. See BatchLoader2 method in EcotypeDefs

            label_dict: dictionary
            Dictionary mapping numeric labels used to train the model to 
            to human-readable labels. This should contain all labels in the
            origional training dataset E.g. 
            label_dict = dict(zip(annot_test['label'], annot_test['Labels']))
            
        Methods
        -------
        evaluate_model()
            Creates model predictions for all evaluation methods. Run first.
        confusion_matrix()
            Creates confusion matrix based on predicted scores and labels. 
        score_distributions()
            Creates violin plots of score distributions for true and false
            positive predictions
        """
        self.model = loaded_model
        self.val_batch_loader = val_batch_loader
        self.label_dict = label_dict
        self.y_true_accum = []
        self.y_pred_accum = []
        self.score_accum = []

    def evaluate_model(self):
        """Runs the model on the validation data and stores predictions and scores."""
        total_batches = len(self.val_batch_loader)
        for i in range(total_batches):
            batch_data = self.val_batch_loader.__getitem__(i)
            batch_scores = self.model.predict(np.asarray(batch_data[0]))  # Model outputs (softmax scores)
            batch_pred_labels = np.argmax(batch_scores, axis=1)
            batch_true_labels = batch_data[1]

            # Accumulate true labels, predicted labels, and scores
            self.y_true_accum.extend(batch_true_labels)
            self.y_pred_accum.extend(batch_pred_labels)
            self.score_accum.extend(batch_scores)

            print(f'Batch {i+1}/{total_batches} processed')

        self.y_true_accum = np.array(self.y_true_accum)
        self.y_pred_accum = np.array(self.y_pred_accum)
        self.score_accum = np.array(self.score_accum)

    def confusion_matrix(self):
        """Computes a confusion matrix with human-readable labels and accuracy."""
        conf_matrix_raw = confusion_matrix(self.y_true_accum, self.y_pred_accum)

        # Normalize confusion matrix by rows
        conf_matrix_percent = conf_matrix_raw.astype(np.float64)
        row_sums = conf_matrix_raw.sum(axis=1, keepdims=True)
        conf_matrix_percent = np.divide(conf_matrix_percent, row_sums, where=row_sums != 0) * 100

        # Map numeric labels to human-readable labels
        unique_labels = sorted(set(self.y_true_accum) | set(self.y_pred_accum))
        human_labels = [self.label_dict[label] for label in unique_labels]

        # Format confusion matrix to two decimal places
        conf_matrix_percent_formatted = np.array([[f"{value:.2f}" for value in row]
                                                  for row in conf_matrix_percent])

        # Create DataFrame
        conf_matrix_df = pd.DataFrame(conf_matrix_percent_formatted, index=human_labels, columns=human_labels)

        # Compute overall accuracy
        accuracy = accuracy_score(self.y_true_accum, self.y_pred_accum)

        return conf_matrix_df, conf_matrix_raw, accuracy

    def score_distributions(self):
        """Generates a DataFrame of score distributions for true positives and false positives."""
        score_data = []
        for i, true_label in enumerate(self.y_true_accum):
            pred_label = self.y_pred_accum[i]
            scores = self.score_accum[i]

            for class_label, score in enumerate(scores):
                label_type = "True Positive" if (true_label == class_label == pred_label) else "False Positive"
                score_data.append({
                    "Class": self.label_dict[class_label],
                    "Score": score,
                    "Type": label_type
                })

        score_df = pd.DataFrame(score_data)

        # Plot paired violin plot
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
    def precision_recall_curves(self):
        """Computes and plots precision-recall curves for all classes."""
        num_classes = self.score_accum.shape[1]
        precision_recall_data = {}
    
        plt.figure(figsize=(10, 8))
    
        # Calculate PR curves for each class
        for class_idx in range(num_classes):
            # Check if the class is present in the dataset
            class_present = (self.y_true_accum == class_idx).any()
    
            if not class_present:
                print(f"Class {self.label_dict[class_idx]} is not present in the validation dataset.")
                # Store empty results for missing class
                precision_recall_data[self.label_dict[class_idx]] = {
                    "precision": None,
                    "recall": None,
                    "average_precision": None
                }
                continue
    
            # Binarize true labels for the current class
            true_binary = (self.y_true_accum == class_idx).astype(int)
    
            # Retrieve scores for the current class
            class_scores = self.score_accum[:, class_idx]
    
            # Compute precision, recall, and average precision score
            precision, recall, _ = precision_recall_curve(true_binary, class_scores)
            avg_precision = average_precision_score(true_binary, class_scores)
    
            # Store the data
            precision_recall_data[self.label_dict[class_idx]] = {
                "precision": precision,
                "recall": recall,
                "average_precision": avg_precision
            }
    
            # Plot PR curve
            plt.plot(recall, precision, label=f"{self.label_dict[class_idx]} (AP={avg_precision:.2f})")
    
        # Finalize plot
        plt.title("Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.show()
    
        return precision_recall_data





#########################################################################
#Pipeline for producing streaming detections- Does not work on GS cloud so comment
##########################################################################


import soundfile as sf

# class AudioProcessor4:
#     def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
#                   params=None, model=None, detection_thresholds=None,
#                   selection_table_name="detections.txt", class_names=None,
#                   table_type="selection"):
#         print('This version is depreciated because it is too damn slow. Use Audiopressor instead')
#         self.folder_path = folder_path
#         self.audio_files = self.find_audio_files(folder_path) if folder_path else []
#         self.segment_duration = segment_duration
#         self.overlap = overlap
#         self.params = params if params else {
#             'outSR': 16000,
#             'clipDur': segment_duration,
#             'nfft': 512,
#             'hop_length': 3200,
#             'spec_type': 'mel',
#             'rowNorm': True,
#             'colNorm': True,
#             'rmDCoffset': True,
#             'inSR': None
#         }
#         self.model = model
#         self.model_input_shape = model.input_shape[1:] if model else None
        
#         self.detection_thresholds = detection_thresholds if detection_thresholds else {
#             class_id: 0.5 for class_id in range(7)
#         }
#         self.class_names = class_names if class_names else {
#             0: 'Abiotic',
#             1: 'BKW',
#             2: 'HW',
#             3: 'NRKW',
#             4: 'Offshore',
#             5: 'SRKW',
#             6: 'Und Bio',
#         }
#         self.selection_table_name = selection_table_name
#         self.table_type = table_type
#         self.init_selection_table()
#         self.detection_counter = 0
        

#     def find_audio_files(self, folder_path):
#         return [os.path.join(root, file)
#                 for root, _, files in os.walk(folder_path)
#                 for file in files if file.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

#     def init_selection_table(self):
#         with open(self.selection_table_name, 'w') as f:
#             if self.table_type == "selection":
#                 f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\n")
#             elif self.table_type == "sound":
#                 f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\tSound\n")

#     def load_audio(self, filename):
#         y, sr = sf.read(filename, dtype='float32')
#         self.params['inSR'] = sr
#         return y, sr

#     def create_segments(self, y, sr):
#         segment_length = int(self.segment_duration * sr)
#         overlap_length = int(self.overlap * sr)
#         start_time = 0.0

#         for start in range(0, len(y) - segment_length + 1, segment_length - overlap_length):
#             yield y[start:start + segment_length], start_time
#             start_time += (segment_length - overlap_length) / sr

#     def create_spectrogram(self, y):
#         expected_time_steps = self.model_input_shape[1] if self.model_input_shape else None
#         spectrogram = create_spectrogram(y, return_snr=False, **self.params)
#         if expected_time_steps and spectrogram.shape[1] < expected_time_steps:
#             spectrogram = np.pad(spectrogram, 
#                                   ((0, 0), 
#                                   (0, expected_time_steps - spectrogram.shape[1])), 
#                                   ymode='constant')
#         elif expected_time_steps and spectrogram.shape[1] > expected_time_steps:
#             spectrogram = spectrogram[:, :expected_time_steps]
#         return spectrogram

#     def process_segment(self, segment, sr, start_time, filename, filestreamStart):
#         spectrogram = self.create_spectrogram(segment)
#         spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
#         predictions = self.model.predict(spectrogram)[0]  # Assuming batch size 1

#         for class_id, score in enumerate(predictions):
#             if score >= self.detection_thresholds[class_id]:
#                 self.output_detection(class_id, score, 
#                                       start_time+filestreamStart, 
#                                       start_time + self.segment_duration+filestreamStart,
#                                       filename)

#     def output_detection(self, class_id, score, start_time, end_time, filename):
#         self.detection_counter += 1
#         with open(self.selection_table_name, 'a') as f:
#             selection = self.detection_counter
#             class_name = self.class_names[class_id]
#             if self.table_type == "selection":
#                 f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\n")
#             elif self.table_type == "sound":
#                 f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\t{filename}\n")

#     def process_all_files(self):
#         filestreamStart = 0
#         for filename in self.audio_files:
#             print(f"Processing file: {filename}")
#             y, sr = self.load_audio(filename)
            
#             for segment, start_time in self.create_segments(y, sr):
#                 self.process_segment(segment, sr, start_time, filename, filestreamStart)
            
#             filestreamStart = filestreamStart+len(y)/sr




class AudioProcessor5:    
    def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
                  params=None, model=None, detection_thresholds=None,
                  selection_table_name="detections.txt", class_names=None,
                  table_type="selection",outputAllScores = False,
                  retain_detections = True):
        """

        Parameters
        ----------
        folder_path : TYPE, optional
            DESCRIPTION. The default is None.
        segment_duration : TYPE, optional
            DESCRIPTION. The default is 2.0.
        overlap : Float, optional
            DESCRIPTION. Seconds of overlap in the audio advancement.
            The default is 1.0.
        params : TYPE, optional
            DESCRIPTION. The default is None.
        model : TYPE, optional
            DESCRIPTION. The default is None.
        detection_thresholds : TYPE, optional
            DESCRIPTION. The default is None.
        selection_table_name : TYPE, optional
            DESCRIPTION. The default is "detections.txt".
        class_names : TYPE, optional
            DESCRIPTION. The default is None.
        table_type : TYPE, optional
            DESCRIPTION. The default is "selection".

        Returns
        -------
        None.

        """
        self.folder_path = folder_path
        self.audio_files = self.find_audio_files(folder_path) if folder_path else []
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.params = params if params else {
            'outSR': 16000,
            'clipDur': segment_duration,
            'nfft': 512,
            'hop_length': 3200,
            'spec_type': 'mel',
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None
        }
        self.model = model
        self.model_input_shape = model.input_shape[1:] if model else None
        self.detection_thresholds = detection_thresholds if detection_thresholds else {
            class_id: 0.5 for class_id in range(7)
        }
        self.class_names = class_names if class_names else {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio',
        }
        self.selection_table_name = selection_table_name
        self.table_type = table_type
        self.init_selection_table()
        self.detection_counter = 0
        self._spec_buffer = None  # Buffer for spectrogram optimization
        self.DataSR = 96000
        self.outputAllScores = False
        self.retain_detections = retain_detections
        self.detections = [] if retain_detections else None

    def find_audio_files(self, folder_path):
        return [os.path.join(root, file)
                for root, _, files in os.walk(folder_path)
                for file in files if file.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    def init_selection_table(self):
        with open(self.selection_table_name, 'w') as f:
            if self.table_type == "selection":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\n")
            elif self.table_type == "sound":
                f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\tScore\tSound\n")

    def load_audio_chunk(self, filename, chunk_size):
        with sf.SoundFile(filename) as sf_file:
            self.DataSR = sf_file.samplerate
            self.params['inSR'] = self.DataSR
            while True:
                y = sf_file.read(frames=chunk_size, dtype='float32', always_2d=False)
                if len(y) == 0:
                    break
                yield y, self.DataSR

    def create_segments_streaming(self, y, sr):
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        
        start = 0
        while start + segment_length <= len(y):
            yield y[start:start + segment_length], start / sr
            start += segment_length - overlap_length

    def create_spectrogram(self, y):
        if self._spec_buffer is None:  # Check if buffer is None (instead of directly checking the array)
            self._spec_buffer = np.zeros((self.params['nfft'] // 2 + 1, self.model_input_shape[1]), dtype=np.float32)
        
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        expected_time_steps = self.model_input_shape[1]
        
        if spectrogram.shape[1] < expected_time_steps:
            self._spec_buffer[:, :spectrogram.shape[1]] = spectrogram
            return self._spec_buffer
        return spectrogram[:, :expected_time_steps]

    def process_batch(self, batch_segments, batch_start_times, batch_files):
        batch_segments = np.stack(batch_segments)  # Create a batch array
        predictions = self.model.predict(batch_segments)  # Model predictions in a batch

        # Two options for output, kick out all the scores (untraditional) or 
        # kick out the scores for the argmax, more traditional
        if self.outputAllScores == False:
            
            # Convert to a numerical array
            numerical_predictions = np.round(np.array(predictions.tolist()), 3)
            
            for i, row in enumerate(numerical_predictions):
                row = np.array(row, dtype=float)  # Convert the row to a numerical array
                max_class = np.argmax(row)
                max_score = np.max(row)
                startTime = batch_start_times[i]
                stopTime = batch_start_times[i] + self.segment_duration
                
                if max_score >= self.detection_thresholds[max_class]:
                    self.output_detection(
                        class_id =max_class, 
                        score= max_score,   
                        start_time= startTime, 
                        end_time= stopTime,
                        filename =batch_files[i]
                    )
                    
        else: # write out all of the scores above the detection threshold
            for i, prediction in enumerate(predictions):
                for class_id, score in enumerate(prediction):
                    if score >= self.detection_thresholds[class_id]:
                        self.output_detection(
                            class_id, score, 
                            batch_start_times[i], 
                            batch_start_times[i] + self.segment_duration,
                            batch_files[i]
                        )
             
            

    def output_detection(self, class_id, score, start_time, end_time, filename):
        
        selection = self.detection_counter
        class_name = self.class_names[class_id]
        
        self.detection_counter += 1
        if self.retain_detections:
               detection = {
                   "Selection": self.detection_counter + 1,
                   "View": "Spectrogram",
                   "Channel": 1,
                   "Begin Time (S)": start_time,
                   "End Time (S)": end_time,
                   "Low Freq (Hz)": 0,
                   "High Freq (Hz)": 8000,
                   "Class": class_name,
                   "Score": score,
                   "File": filename}
               self.detections.append(detection)
        
        with open(self.selection_table_name, 'a') as f:
            if self.table_type == "selection":
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\n")
            elif self.table_type == "sound":
                f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\t{score:.4f}\t{filename}\n")

    def get_detections(self, as_dataframe=True):
        if not self.retain_detections:
            raise ValueError("Retention of detections was not enabled. Set retain_detections=True in the constructor.")
        return pd.DataFrame(self.detections) if as_dataframe else np.array(self.detections)
    
    def process_all_files(self):
        filestreamStart = 0  # Initialize the global start time
    
        for filename in self.audio_files:
            print(f"Processing file: {filename}")
            chunk_start_time = 0  # Initialize chunk start time for each file
            batch_segments, batch_start_times, batch_files = [], [], []
    
            # Process each audio chunk
            for audio_chunk, sr in self.load_audio_chunk(filename, chunk_size=self.DataSR * 15):  # Process 15-second chunks
                for segment, start_time in self.create_segments_streaming(audio_chunk, self.DataSR):
                    # Adjust the segment's start time to be relative to the whole filestream
                    global_start_time = filestreamStart + chunk_start_time + start_time
                    batch_segments.append(self.create_spectrogram(segment))
                    batch_start_times.append(global_start_time)
                    batch_files.append(filename)
    
                    # If batch size is reached, process the batch
                    if len(batch_segments) == 32:
                        self.process_batch(batch_segments, batch_start_times, batch_files)
                        batch_segments, batch_start_times, batch_files = [], [], []
    
                # Update chunk_start_time for the next chunk
                chunk_start_time += len(audio_chunk) / sr
    
            # Process remaining segments if any
            if batch_segments:
                self.process_batch(batch_segments, batch_start_times, batch_files)
    
            # Update the global filestreamStart after processing the current file
            filestreamStart += chunk_start_time  # Increment global filestream start time




    
# if __name__ == "__main__":
    
#     # Example how to call
    
#     # Load Keras model
#     folder_path = 'E:\\Malahat\\STN3\\20151028'
    
#     # Load Keras model
#     model_path = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Models\\20200818\\output_resnet18_PECN_melSpec.keras'
#     model = load_model(model_path)

#     # Spectrogram parameters
#     AudioParms = {
#         'clipDur': 2,
#         'outSR': 16000,
#         'nfft': 512,
#         'hop_length': 25,
#         'spec_type': 'mel',  
#         'spec_power': 2,
#         'rowNorm': False,
#         'colNorm': False,
#         'rmDCoffset': False,
#         'inSR': None, 
#         'PCEN': True,
#         'fmin': 150
#     }

#     # Example detection thresholds (adjust as needed)
#     detection_thresholds = {
#         0: 0.8,  # Example threshold for class 0
#         1: 0.8,  # Example threshold for class 1
#         2: 0.9,  # Example threshold for class 2
#         3: 0.8,  # Example threshold for class 3
#         4: 0.8,  # Example threshold for class 4
#         5: 0.8,  # Example threshold for class 5
#         6: 0.9   # Example threshold for class 6
#     }

#     class_names = {
#         0: 'Abiotic',
#         1: 'BKW',
#         2: 'HW',
#         3: 'NRKW',
#         4: 'Offshore',
#         5: 'SRKW',
#         6: 'Und Bio'
#     }

    
    
#     # Initialize the AudioProcessor with your model and detection thresholds
#     processor = AudioProcessor2(folder_path=folder_path, model=model,
#                                detection_thresholds=detection_thresholds, 
#                                class_names=class_names,
#                                params= AudioParms,
#                                table_type="sound",
#                                overlap=0.25)
    
#     # Process all audio files in the directory
#     processor.process_all_files()




