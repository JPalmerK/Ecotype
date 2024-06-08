# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:28:09 2024

@author: kaity
"""

#####################################################################
# Use the HDF5 and batch laoder to train a simple CNN
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Input
from keras import Model
from tqdm import tqdm
import random
import h5py


train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/AllAnno4khz_Melint16.h5'

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
num_classes =7

# Create and compile model
input_shape_with_channels = input_shape + (1,)  # Add channel
model = create_model(input_shape_with_channels, num_classes)
model = compile_model(model)

# Train model
train_model(model, batch_loader, epochs=3, num_classes=num_classes)



# Evaluate model
batch_loader = BatchLoader(train_hdf5_file, batch_size=32)
test_data, test_labels = batch_loader.get_test_batch()
accuracy = model.evaluate(test_data, test_labels)[1]
print("Test Accuracy:", accuracy)







# Usage example


train_shapes = check_spectrogram_dimensions(train_hdf5_file)

print("Unique shapes of spectrograms in train dataset:", train_shapes)
print("Unique shapes of spectrograms in test dataset:", test_shapes)