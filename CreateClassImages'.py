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
###################################################################


# Save data to HDF5 file
# Function to process data and store it in HDF5 file
def process_and_store_data(AllAnno, hdf5_file):
    with h5py.File(hdf5_file, 'w') as hf:
        
        trainTestCounts =AllAnno['traintest'].value_counts()
        trainIdx =0
        testIdx=0
        
        train_group = hf.create_group('train')
        test_group = hf.create_group('test')
        

        n_rows = len(AllAnno)
        # Read CSV file and process data
        # Replace this with your code to iterate through the CSV file and generate spectrogram clips
        #for index, row in AllAnno.iterrows():
        for ii in range(1, n_rows):
            row = AllAnno.iloc[ii]
            file_path = row['FilePath']
            start_time = row['FileBeginSec']
            end_time = row['FileEndSec']
            label = row['label']
            traintest = row['traintest']
                
            if not np.isnan(row.FileEndSec):
                # Load and process audio segment
                spec_normalized = load_and_process_audio_segment(
                    file_path, start_time, end_time)
                
            
                # Make it smaller! 
                spec_normalized = (spec_normalized * 65535).astype(np.uint16)
                
                # Split data into train and test 
                if traintest == 'Train':
                    group = train_group
                    val = trainIdx
                    trainIdx= trainIdx+1
                else:
                    group = test_group
                    val = testIdx
                    testIdx= testIdx+1
                    
                print(ii, ' of ', len(AllAnno))
                # Create dataset
                group.create_dataset(f'spectrogram_{val}', data=spec_normalized)
                group.create_dataset(f'label_{val}', data=label)





train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/AllAnno4khz_Mel10.h5'
process_and_store_data(AllAnno=AllAnno.iloc[0:999], hdf5_file= train_hdf5_file)




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

train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/AllAnno4khz_Mel13.h5'
create_hdf5_dataset(annotations=AllAnno.iloc[0:100], hdf5_filename= train_hdf5_file)



# Batch loader for HDF5 dataset

class BatchLoader:
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
            if self.test_index < self.num_test_samples:
                key = self.test_keys[self.test_index]
                spec = self.hf['test'][key]['spectrogram'][:]
                label = self.hf['test'][key]['label'][()]  # Access scalar value
                batch_data.append(spec)
                batch_labels.append(label)
                self.test_index += 1

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

# Example usage
batch_loader = BatchLoader(train_hdf5_file, batch_size=5)
train_data, train_labels = batch_loader.get_train_batch()
test_data, test_labels = batch_loader.get_test_batch()












hf = h5py.File(train_hdf5_file, 'r')
hf['train']['spectrograms'][1:3,:,:]

# Get a list of dataset names within the group
group = hf['train']['spectrograms']
dataset_names = list(group.keys())


##


import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load data from HDF5 file in chunks
def load_data_in_chunks(hdf5_file, chunk_size=1000):
    with h5py.File(hdf5_file, 'r') as hf:
        n_samples = len(hf['train']['spectrograms'])
        for i in range(0, n_samples, chunk_size):
            spectrograms_chunk = np.array(hf['train']['spectrograms'][i:i+chunk_size])
            labels_chunk = np.array(hf['train']['labels'][i:i+chunk_size])
            yield spectrograms_chunk, labels_chunk
            
            
            



X_train = np.concatenate(X_train)
X_val = np.concatenate(X_val)
y_train = np.concatenate(y_train)
y_val = np.concatenate(y_val)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 314, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/test_data.h5'
test_spectrograms, test_labels = [], []
for test_spectrograms_chunk, test_labels_chunk in load_data_in_chunks(test_hdf5_file):
    test_spectrograms_chunk = test_spectrograms_chunk.reshape(test_spectrograms_chunk.shape[0], 128, 314, 1)
    test_spectrograms.append(test_spectrograms_chunk)
    test_labels.append(test_labels_chunk)

test_spectrograms = np.concatenate(test_spectrograms)
test_labels = np.concatenate(test_labels)

loss, accuracy = model.evaluate(test_spectrograms, test_labels)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)












testIdx=100
file_path = AllAnno['FilePath'].iloc[testIdx]  # Path to your audio file
start_time = AllAnno['FileBeginSec'].iloc[testIdx]  # Start time in seconds
duration = AllAnno['FileEndSec'].iloc[testIdx]-AllAnno['FileBeginSec'].iloc[testIdx]  # Duration of the audio section in seconds
end_time= AllAnno['FileEndSec'].iloc[testIdx]

# Test the data production
spec_img = load_and_process_audio_segment(file_path, 
                                                           start_time, 
                                                           end_time,clipDur =2)

spectrogram_data_16bit = (normxpec * 65535).astype(np.uint16)
spectrogram_data_8bit = (normxpec * 255).astype(np.uint8)
# # Create directories to save spectrogram images
# train_dir = "C:/Users/kaity/Documents/GitHub/Ecotype/train_images"
# test_dir = "C:/Users/kaity/Documents/GitHub/Ecotype/test_images"
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)







