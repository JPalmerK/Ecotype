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


annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest.csv")
AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 4000]

AllAnno = AllAnno.dropna(subset=['FileEndSec'])
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)



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
    
    # spec_img = librosa.display.specshow(spec_normalized,
    #                                       sr=sample_rate,
    #                                       hop_length=hop_length,     
    #                                       x_axis='time',
    #                                       y_axis='log')
    
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram')
    
    # plt.tight_layout()
    # plt.show()
    
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

            spec = load_and_process_audio_segment(file_path, start_time, end_time)

            if traintest == 'Train':  # 80% train, 20% test
                dataset = train_group
            else:
                dataset = test_group

            dataset.create_dataset(f'{ii}/spectrogram', data=spec)
            dataset.create_dataset(f'{ii}/label', data=label)
            print(ii, ' of ', len(annotations))

train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/AllAnno4khz_Mel17.h5'
create_hdf5_dataset(annotations=AllAnno.iloc[0:5000], hdf5_filename= train_hdf5_file)

hf = h5py.File(train_hdf5_file, 'r')

# Batch loader for HDF5 dataset

class BatchLoader:
    def __init__(self, hdf5_file, batch_size):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.train_keys = list(self.hf['train'].keys())
        self.test_keys = list(self.hf['test'].keys())
        self.num_train_samples = len(self.train_keys)
        self.num_test_samples = len(self.test_keys)

    def get_train_batch(self):
        batch_data = []
        batch_labels = []

        train_indices = random.sample(range(self.num_train_samples), self.batch_size)
        for index in train_indices:
            key = self.train_keys[index]
            spec = self.hf['train'][key]['spectrogram'][:]
            label = self.hf['train'][key]['label'][()]  # Access scalar value
            batch_data.append(spec)
            batch_labels.append(label)

        return np.array(batch_data), np.array(batch_labels)

    def get_test_batch(self):
        batch_data = []
        batch_labels = []

        test_indices = random.sample(range(self.num_test_samples), self.batch_size)
        for index in test_indices:
            key = self.test_keys[index]
            spec = self.hf['test'][key]['spectrogram'][:]
            label = self.hf['test'][key]['label'][()]  # Access scalar value
            batch_data.append(spec)
            batch_labels.append(label)

        return np.array(batch_data), np.array(batch_labels)
# Example usage
batch_loader = BatchLoader(train_hdf5_file, batch_size=32)
train_data, train_labels = batch_loader.get_train_batch()
test_data, test_labels = batch_loader.get_test_batch()

    def __init__(self, hdf5_file, batch_size):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.train_keys = list(self.hf['train'].keys())
        self.test_keys = list(self.hf['test'].keys())
        self.num_train_samples = len(self.train_keys)
        self.num_test_samples = len(self.test_keys)
        self.train_index = 0
        self.test_index = 0

    def get_train_batch(self):
        batch_data = []
        batch_labels = []

        for _ in range(self.batch_size):
            if self.train_index >= self.num_train_samples:
                self.train_index = 0
            key = self.train_keys[self.train_index]
            spec = self.hf['train'][key]['spectrogram'][:]
            label = self.hf['train'][key]['label'][()]  # Access scalar value
            batch_data.append(spec)
            batch_labels.append(label)
            self.train_index += 1

        return np.array(batch_data), np.array(batch_labels)

    def get_test_batch(self):
        batch_data = []
        batch_labels = []

        for _ in range(self.batch_size):
            if self.test_index >= self.num_test_samples:
                self.test_index = 0
            key = self.test_keys[self.test_index]
            spec = self.hf['test'][key]['spectrogram'][:]
            label = self.hf['test'][key]['label'][()]  # Access scalar value
            batch_data.append(spec)
            batch_labels.append(label)
            self.test_index += 1

        return np.array(batch_data), np.array(batch_labels)
#####################################################################
# Use the HDF5 and batch laoder to train a simple CNN
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Input
from keras import Model
from tqdm import tqdm
# Define CNN model

def create_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Compile model
def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model
# Train model
def train_model(model, batch_loader, epochs, num_classes):
    for epoch in range(epochs):
        total_train_batches = batch_loader.num_train_samples // batch_loader.batch_size
        total_test_batches = batch_loader.num_test_samples // batch_loader.batch_size
        
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        
        # Train loop with progress bar
        with tqdm(total=total_train_batches, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for _ in range(total_train_batches):
                train_data, train_labels = batch_loader.get_train_batch()
                train_data = np.expand_dims(train_data, axis=-1)  # Add channel dimension
                train_labels = to_categorical(train_labels, num_classes=num_classes)
                
                loss, accuracy = model.train_on_batch(train_data, train_labels)
                train_loss += loss
                train_accuracy += accuracy
                pbar.update(1)
        
        # Test loop
        for _ in range(total_test_batches):
            test_data, test_labels = batch_loader.get_test_batch()
            test_data = np.expand_dims(test_data, axis=-1)  # Add channel dimension
            test_labels = to_categorical(test_labels, num_classes=num_classes)
            loss, accuracy = model.test_on_batch(test_data, test_labels)
            test_loss += loss
            test_accuracy += accuracy
        
        train_loss /= total_train_batches
        train_accuracy /= total_train_batches
        test_loss /= total_test_batches
        test_accuracy /= total_test_batches
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    for epoch in range(epochs):
        total_train_batches = batch_loader.num_train_samples // batch_loader.batch_size
        total_test_batches = batch_loader.num_test_samples // batch_loader.batch_size
        
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        
        # Train loop
        for _ in range(total_train_batches):
            train_data, train_labels = batch_loader.get_train_batch()
            train_data = np.expand_dims(train_data, axis=-1)  # Add channel dimension
            train_labels = to_categorical(train_labels, num_classes=num_classes)
            
            loss, accuracy = model.train_on_batch(train_data, train_labels)
            train_loss += loss
            train_accuracy += accuracy
        
        # Test loop
        for _ in range(total_test_batches):
            test_data, test_labels = batch_loader.get_test_batch()
            test_data = np.expand_dims(test_data, axis=-1)  # Add channel dimension
            test_labels = to_categorical(test_labels, num_classes=num_classes)
            loss, accuracy = model.test_on_batch(test_data, test_labels)
            test_loss += loss
            test_accuracy += accuracy
        
        train_loss /= total_train_batches
        train_accuracy /= total_train_batches
        test_loss /= total_test_batches
        test_accuracy /= total_test_batches
        
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    for epoch in range(epochs):
        total_train_batches = batch_loader.num_train_samples // batch_loader.batch_size
        total_test_batches = batch_loader.num_test_samples // batch_loader.batch_size
        
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        
        # Train loop
        for _ in range(total_train_batches):
            train_data, train_labels = batch_loader.get_train_batch()
            train_data = np.expand_dims(train_data, axis=-1)  # Add channel dimension
            train_labels = to_categorical(train_labels, num_classes=num_classes)
            
            loss, accuracy = model.train_on_batch(train_data, train_labels)
            train_loss += loss
            train_accuracy += accuracy
        
        # Test loop
        for _ in range(total_test_batches):
            test_data, test_labels = batch_loader.get_test_batch()
            test_data = np.expand_dims(test_data, axis=-1)  # Add channel dimension
            test_labels = to_categorical(test_labels, num_classes=num_classes)
            loss, accuracy = model.test_on_batch(test_data, test_labels)
            test_loss += loss
            test_accuracy += accuracy
        
        train_loss /= total_train_batches
        train_accuracy /= total_train_batches
        test_loss /= total_test_batches
        test_accuracy /= total_test_batches
        
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Initialize batch loader
batch_loader = BatchLoader(train_hdf5_file, batch_size=32)



# Get input shape and number of classes
input_shape = batch_loader.get_train_batch()[0].shape[1:]  # Shape of a single spectrogram
input_shape_with_channels = input_shape + (1,)  # Add channel dimension
num_classes = len(np.unique(AllAnno['Labels']))

# Create and compile model
model = create_model(input_shape_with_channels, num_classes)
model = compile_model(model)

# Train model
train_model(model, batch_loader, epochs=5, num_classes=num_classes)






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


train_shapes = check_spectrogram_dimensions(train_hdf5_file)

print("Unique shapes of spectrograms in train dataset:", train_shapes)
print("Unique shapes of spectrograms in test dataset:", test_shapes)