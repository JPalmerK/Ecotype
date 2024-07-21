# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 06:12:13 2024

@author: kaity
"""

# Back to basics, 7 classes no culling

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model

from tensorflow.keras.callbacks import TensorBoard

# Define the train and test batch loader
# Example usage



train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.h5'
train_batch_loader = Eco.BatchLoader2(train_hdf5_file, 
                           trainTest = 'train', batch_size=250, n_classes=7,
                           minFreq=0)
test_batch_loader =  Eco.BatchLoader2(train_hdf5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=7,
                           minFreq=0)

# Create the Resnet 
num_classes =7
test_batch_loader.specSize
input_shape = (128, 314, 1)

# Create and compile model
smallResnetBatchNorm = Eco.create_model_with_resnet(input_shape, num_classes, 
                                               actName = 'relu')
smallResnetBatchNorm = Eco.compile_model(smallResnetBatchNorm)


# Define a directory to store TensorBoard logs
log_dir = "logs/smallResnetBatchNorm/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Train the model
Eco.train_model(smallResnetBatchNorm, train_batch_loader,
            test_batch_loader, epochs=20)
smallResnetBatchNorm.save('C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_7class_8khz_batchnorm.keras')


# Evaluate the model
ValData = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatVal_08_khz_Melint16MetricsSNR.h5'
val_batch_loader =  Eco.BatchLoader2(ValData, 
                           trainTest = 'train', batch_size=1000,  
                           n_classes=7,   minFreq=0 )

confResults = Eco.batch_conf_matrix(loaded_model= smallResnetBatchNorm, 
                                val_batch_loader=val_batch_loader)


# Ok, adding batch normalization did not seem to help.
##############################################################################
# Create a slightly larger resnet
#############################################################################

# Create and compile model
import EcotypeDefs as Eco
from keras.models import load_model

train_hdf5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.h5'
train_batch_loader = Eco.BatchLoader2(train_hdf5_file, 
                           trainTest = 'train', batch_size=128, n_classes=7,
                           minFreq=0)
test_batch_loader =  Eco.BatchLoader2(train_hdf5_file, 
                           trainTest = 'test', batch_size=128,  n_classes=7,
                           minFreq=0)

num_classes =7
test_batch_loader.specSize
input_shape = (128, 314, 1)

widerResnet = Eco.create_wider_model2(input_shape, num_classes, actName='relu')
widerResnet = Eco.compile_model(widerResnet)


# Train the model- trained three epocs. Retrain tonight another few.
Eco.train_model(widerResnet, train_batch_loader, test_batch_loader, epochs=12)
widerResnet.save('C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_7class_8khz_wider_5.keras')
widerResnet = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_7class_8khz_wider_3.keras')

# Evaluate the model
ValData = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatVal_08_khz_Melint16MetricsSNR.h5'
val_batch_loader =  Eco.BatchLoader2(ValData, 
                           trainTest = 'train', batch_size=1000,  
                           n_classes=7,   minFreq=0 )

confResults = Eco.batch_conf_matrix(loaded_model= widerResnet, 
                                val_batch_loader=val_batch_loader)




