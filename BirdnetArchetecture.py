# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:09:32 2025

@author: kaity
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def residual_block(x, filters):
    """ Residual block: Conv + BN + ReLU + Conv + BN + Add """
    shortcut = x  # Identity mapping
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Residual connection
    x = layers.ReLU()(x)
    return x

def downsampling_block(x, filters):
    """ Downsampling block: Conv (stride=2) + BN + ReLU """
    x = layers.Conv2D(filters, (3, 3), strides=2, padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def create_birdnet(input_shape=(64, 384, 1), num_classes=987):
    inputs = layers.Input(shape=input_shape)

    # Preprocessing Conv Block
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)

    # ResStack 1
    x = downsampling_block(x, 64)   # (64, 32, 96)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # ResStack 2
    x = downsampling_block(x, 128)  # (128, 16, 48)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # ResStack 3
    x = downsampling_block(x, 256)  # (256, 8, 24)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # ResStack 4
    x = downsampling_block(x, 512)  # (512, 4, 12)
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    # Classification Head
    x = layers.Conv2D(512, (4, 10), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(1024, (1, 1), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(num_classes, (1, 1), padding="same", activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Global Log-Mel Energy Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output Layer with Sigmoid Activation
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)  

    model = models.Model(inputs, outputs, name="BirdNET")
    return model

# Compile model
def compile_model(model):
    model.compile(
        loss='binary_crossentropy',  # Multi-label classification
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
        metrics=['accuracy']
    )
    return model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Define custom callback for dropout adjustment
class DropoutScheduler(tf.keras.callbacks.Callback):
    """Reduces dropout probability by 0.1 when validation loss stalls."""
    def __init__(self, model, factor=0.1, min_dropout=0.1, patience=3):
        super(DropoutScheduler, self).__init__()
        self.network = model  # Renamed to avoid conflict
        self.factor = factor
        self.min_dropout = min_dropout
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return
        
        # If validation loss has stalled
        if val_loss >= self.best_loss:
            self.wait += 1
            if self.wait >= self.patience:
                self.adjust_dropout()
                self.wait = 0
        else:
            self.best_loss = val_loss
            self.wait = 0  # Reset patience if improvement happens

    def adjust_dropout(self):
        for layer in self.network.layers:  # Use self.network instead
            if isinstance(layer, tf.keras.layers.Dropout):
                new_rate = max(layer.rate - self.factor, self.min_dropout)
                layer.rate = new_rate  # Update dropout rate
                print(f"Reduced dropout rate to {new_rate}")


# Define training function
def train_birdnet(model, train_data, val_data, initial_lr=0.0005,
                  batch_size=32, epochs=50, weights_path="birdnet_best.h5"):
    """Train BirdNET with step-wise LR reduction, dropout adjustment, and knowledge distillation."""

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)

    dropout_scheduler = DropoutScheduler(model)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, 
                        validation_data=val_data, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        callbacks=[lr_scheduler, early_stopping, 
                                   checkpoint, dropout_scheduler])

    return history

# Load Pre-Trained Weights (Warm Start)
def load_pretrained_model(model, weights_path="birdnet_best.h5"):
    """Loads pre-trained weights to initialize training."""
    try:
        model.load_weights(weights_path)
        print(f"Loaded pre-trained weights from {weights_path}")
    except Exception as e:
        print("No pre-trained weights found, starting from scratch.")

# Knowledge Distillation (Born-Again Network)
def born_again_train(student_model, teacher_model, train_data, 
                     val_data, alpha=0.5, temperature=3, 
                     batch_size=32, epochs=50):
    """Trains a new model using the best previous model as a teacher (Born-Again Network)."""
    
    # Create soft labels from the teacher model
    def distillation_loss(y_true, y_pred):
        soft_targets = teacher_model.predict(train_data)
        soft_loss = tf.keras.losses.KLDivergence()(soft_targets, tf.nn.softmax(y_pred / temperature))
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                          loss=distillation_loss, metrics=['accuracy'])

    history = student_model.fit(train_data, 
                                validation_data=val_data, 
                                epochs=epochs, 
                                batch_size=batch_size,
                                callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
                                           EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, cooldown=3, verbose=1),
                                           ModelCheckpoint("birdnet_born_again.h5", monitor='val_loss', save_best_only=True, verbose=1)])

    return history


import numpy as np
import h5py
import keras

class BatchLoader_hardNegs(keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size=250, trainTest='train',
                 shuffle=True, n_classes=7, return_data_labels=False,
                 minFreq=None, max_samples_per_class=500, teacher_model=None):
        self.hf = h5py.File(hdf5_file, 'r')
        self.batch_size = batch_size
        self.trainTest = trainTest
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.return_data_labels = return_data_labels
        self.max_samples_per_class = max_samples_per_class
        self.teacher_model = teacher_model  # For knowledge distillation
        
        # Store references to both train and test groups
        self.train_group = self.hf['train']
        self.test_group = self.hf['test']
        
        # Get data keys and their labels
        self.data_keys = list(self.hf[trainTest].keys())
        self.key_source = {key: trainTest for key in self.data_keys}  # Track which group a key belongs to
        self.labels = {key: self.hf[trainTest][key]['label'][()] for key in self.data_keys}
        
        # Enforce class balancing
        self._filter_class_samples()
        
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
        
        # Get spectrogram size
        self.first_key = self.data_keys[0]
        self.specSize = self.hf[trainTest][self.first_key]['spectrogram'].shape
        
        self.minFreq = minFreq
        self.minIdx = 0

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _filter_class_samples(self):
        """Limits the number of samples per class."""
        class_counts = {i: 0 for i in range(self.n_classes)}
        filtered_keys = []

        for key in self.data_keys:
            label = self.labels[key]
            if class_counts[label] < self.max_samples_per_class:
                filtered_keys.append(key)
                class_counts[label] += 1
        
        self.data_keys = filtered_keys

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.num_samples)
        
        batch_data = []
        batch_labels = []
        
        for i in range(start_index, end_index):
            key = self.data_keys[self.indexes[i]]
            source_group = self.key_source[key]
            
            spec = self.hf[source_group][key]['spectrogram'][self.minIdx:, :]
            label = self.labels[key]
            batch_data.append(spec)
            
            if self.teacher_model:
                # Generate soft labels using the teacher model
                soft_label = self.teacher_model.predict(np.expand_dims(spec, axis=0))[0]
                batch_labels.append(soft_label)
            else:
                batch_labels.append(keras.utils.to_categorical(label, num_classes=self.n_classes))
        
        return np.array(batch_data), np.array(batch_labels)

    def add_hard_negatives(self, hard_negatives):
        """Move hard negatives from test to train and update indices."""
        new_keys = [key for key in hard_negatives if key in self.test_group]
        
        if not new_keys:
            print("Warning: No valid hard negatives found in test.")
            return
        
        for key in new_keys:
            self.key_source[key] = 'test'  # Track original source
            self.labels[key] = self.hf['test'][key]['label'][()]
        
        self.data_keys = list(np.unique(self.data_keys + new_keys))
        self._filter_class_samples()  # Reapply class limits
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
        
        self.__shuffle__()
        print(f"Training set now has {self.num_samples} samples (including hard negatives).")

    def increase_sample_limit(self, increment=1000):
        """Increase the max number of samples per class and refresh the dataset."""
        self.max_samples_per_class += increment
        self._filter_class_samples()
        self.num_samples = len(self.data_keys)
        self.indexes = np.arange(self.num_samples)
        print(f"Increased sample limit to {self.max_samples_per_class} per class.")

    def __shuffle__(self):
        np.random.shuffle(self.indexes)
        print('Shuffled!')

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            print('Epoch end: all shuffled!')


if __name__ == "__main__":
    
    # Set up audio parameters
    AudioParms = {
                'clipDur': 3,
                'outSR': 15000,
                'nfft': 512,
                'hop_length':51,
                'spec_type': 'mel',  
                'rowNorm': False,
                'colNorm': False,
                'rmDCoffset': False,
                'inSR': None, 
                'PCEN': True,
                'fmin': 0,
                'min_freq': None,       # default minimum frequency to retain
                'spec_power':1,
                'returnDB':False,         # return spectrogram in linear or convert to db 
                'NormalizeAudio': True,
                'Scale Spectrogram': True,
                'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
                'using batch norm and Raven parameters with Mel Spectrograms and PCEN '} # scale the spectrogram between 0 and 1

    # #%% Make the HDF5 files for training testing and evaluation
    h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/MediumDatacetONCfixed_15khz\\data\\Malahat_MnBalanced_15khz_512fft_PCEN_4k_round.h5'
    h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/MediumDatacetONCfixed_15khz/data/MnBalanced_15khz_512fft_PCEN_4k_round.h5'
    modelSaveLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\MediumDatacetONCfixed_15khz\\MnBalanced_15khz_512fft_PCEN_4k_birdnetLike_negMining.keras'




    # Create the batch loader which will report the size

    valLoader =  BatchLoader_hardNegs(h5_file_validation, 
                                trainTest = 'train', batch_size=200,  n_classes=7,  
                                minFreq=0,   return_data_labels = False)

    trainLoader = BatchLoader_hardNegs(h5_fileTrainTest, 
                                trainTest = 'train', batch_size=64,  n_classes=7,  
                                minFreq=0)
    testLoader =  BatchLoader_hardNegs(h5_fileTrainTest, 
                                trainTest = 'test', batch_size=64,  n_classes=7,  
                                minFreq=0)

    
    testLoader.specSize
    model = create_birdnet(input_shape= (128, 883,1), num_classes=7)
    load_pretrained_model(model)  # Load weights if available
    history = train_birdnet(model, trainLoader, testLoader)
