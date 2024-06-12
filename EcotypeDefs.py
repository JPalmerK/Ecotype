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
    file_duration = librosa.get_duration(filename=file_path)
    
    # Calculate the duration of the audio segment
    duration = end_time - start_time
    
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
    
    return spec_normalized

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

            spec = load_and_process_audio_segment(file_path, start_time, end_time)

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
            
            
            print(ii, ' of ', len(annotations))
    
class BatchLoader2(keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels=return_data_labels
        self.data_keys = list(self.hf[trainTest].keys())
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
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
            spec = self.hf[self.trainTest][key]['spectrogram'][:]
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


# Define the training function that works with the batch generagor
def train_model(model, train_generator, val_generator, epochs):
    # Define early stopping callback
   early_stopping = EarlyStopping(monitor='val_loss', patience=3,
                                  restore_best_weights=True)
   
   # Train the model with early stopping
   model.fit(x=train_generator,
             epochs=epochs,
             validation_data=val_generator,
             callbacks=[early_stopping])
   model.fit(x=train_generator,
              epochs=epochs,
              validation_data=val_generator)

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


def create_model_with_resnet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add three identity blocks
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def identity_block(input_tensor, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Add()([x, input_tensor])  # Add the input tensor to the output of the second convolution
    x = Activation('relu')(x)
    return x




ValData  = 'C:/Users/kaity/Documents/GitHub/Ecotype/Malahat4khz_Melint16.h5'
val_batch_loader =  BatchLoader2(ValData, 
                           trainTest = 'train', batch_size=500,  n_classes=7)

# Load the model from the saved file
loaded_model = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models/Resnetv1.keras')




def batch_conf_matrix(loaded_model, val_batch_loader):
    # Initialize lists to accumulate predictions and true labels
    y_pred_accum = []
    y_true_accum = []
    
    # Get the total number of batches
    total_batches = len(val_batch_loader)
    
    # Iterate over test data generator batches
    for i, (batch_data, batch_labels) in enumerate(val_batch_loader):
        # Predict on the current batch
        batch_pred = loaded_model.predict(batch_data)
        
        # Convert predictions to class labels
        batch_pred_labels = np.argmax(batch_pred, axis=1)
        
        # Convert true labels to class labels
        batch_true_labels = np.argmax(batch_labels, axis=1)
        
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
    
    print("Confusion Matrix:")




if __name__ == "__main__":
    
    
    
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 4000]
AllAnno = AllAnno.dropna(subset=['FileEndSec'])


#AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)
annot_malahat = pd.read_csv('C:/Users/kaity/Documents/GitHub/Ecotype/Malahat.csv')
Validation =annot_malahat[annot_malahat['LowFreqHz'] < 4000]



label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()
label_mapping_Validation = Validation[['label', 'Labels']].drop_duplicates()



