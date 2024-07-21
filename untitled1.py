# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:03:28 2024

@author: kaity
"""
import numpy as np
import random
import h5py
import keras

class BatchLoader:
    def __init__(self, hdf5_file, batch_size, trainTest = 'train',
                 shuffle= True, n_classes=7):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes =n_classes
        self.data_keys = list(self.hf[trainTest].keys())
        self.num_samples = len(self.data_keys)
        self.data_indices = list(range(self.num_samples))
        random.shuffle(self.data_indices)
        self.current_index = 0
        self.epocBatch = np.ceil(self.num_samples / self.batch_size)
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.num_samples) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.data_indices = np.arange(len(self.data_keys))
        if self.shuffle == True:
            np.random.shuffle(self.data_indices)
            
    def get_batch(self):
        batch_data = []
        batch_labels = []

        start_index = self.current_index
        end_index = min(start_index + self.batch_size, self.num_samples)
        for i in range(start_index, end_index):
            index = self.data_indices[i]
            key = self.data_keys[index]
            spec = self.hf[self.trainTest][key]['spectrogram'][:]
            label = self.hf[self.trainTest][key]['label'][()]
            batch_data.append(spec)
            batch_labels.append(label)

        self.current_index = end_index % self.num_samples

        #percentage = (self.current_index / self.num_samples) * 100
        #print(f"Percentage of epoch completed for training: {percentage:.2f}%")

        return np.array(batch_data),  keras.utils.to_categorical(np.array(batch_labels), num_classes=self.n_classes) 

   
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


##########################################################################


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

# Define the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the training function
def train_model(model, train_generator, val_generator, epochs):
    model.fit(x=train_generator,
              epochs=epochs,
              validation_data=val_generator)
# Example usage:
# Assuming input shape (314, 128, 1) for single channel images
# and 7 classes
input_shape = (128, 314, 1)
num_classes = 7
batch_size = 32
epochs = 1

# Create the CNN model
model = create_model(input_shape, num_classes)


# Define the train and test batch loader
# Example usage
train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/AllAnno4khz_Melint16.h5'
train_batch_loader = BatchLoader2(train_hdf5_file, 
                           trainTest = 'train', batch_size=150, n_classes=7)
test_batch_loader =  BatchLoader2(train_hdf5_file, 
                           trainTest = 'test', batch_size=150,  n_classes=7)

# Train the model
train_model(model, train_batch_loader, test_batch_loader, epochs=10)
model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Models/pilot_v1.h5')
model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Models/pilot_v1.keras')

###########################################################################
# Load and evaluate the model
##########################################################################

from keras.models import load_model

test_batch_loader =  BatchLoader2(train_hdf5_file, 
                           trainTest = 'test', batch_size=500,  n_classes=7)


# Load the model from the saved file
loaded_model = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models/pilot_v1.keras')



# Initialize lists to accumulate predictions and true labels
y_pred_accum = []
y_true_accum = []

# Get the total number of batches
total_batches = len(test_batch_loader)


# Iterate over test data generator batches
for i, (batch_data, batch_labels) in enumerate(test_batch_loader):
    # Predict on the current batch
    batch_pred = model.predict(batch_data)
    
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
print(conf_matrix)
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



###########################################################################
# Work someting up more like resnet.
#########################################################################

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Activation

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




# Initialize batch loader


train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/AllAnno4khz_Melint16.h5'
train_batch_loader = BatchLoader2(train_hdf5_file, 
                           trainTest = 'train', batch_size=150, n_classes=7)
test_batch_loader =  BatchLoader2(train_hdf5_file, 
                           trainTest = 'test', batch_size=150,  n_classes=7)


# Get input shape and number of classes
num_classes =7
input_shape = (128, 314, 1)

# Create and compile model
model = create_model_with_resnet(input_shape, num_classes)
model = compile_model(model)

# Train model
train_model(model, train_batch_loader, test_batch_loader, epochs=1)