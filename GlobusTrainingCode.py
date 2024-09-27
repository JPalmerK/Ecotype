# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:13:49 2024

@author: Kait Palmer, kpalmer@coa.edu
"""


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco # custom functions
from keras.models import load_model
import librosa

# Only hard coded once :) 
# Load HDF5 files for train/test and evaluation - to be updated
h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_SNR_PCEN.h5'
h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\MalahatBalanced_melSpec_8khz_PCEN.h5'

# Keras Model output locatoin
modelOutLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_PCEN.keras'





# Parameters for audio represnetations
AudioParms = {
            'clipDur': 2, #seconds
            'outSR': 16000, #exported sample rate
            'nfft': 512,
            'hop_length':25,
            'spec_type': 'mel',  # options, mel or 'normal'
            'spec_power':2, # for GPL like power law
            'rowNorm': False, # normalize by removing row medians
            'colNorm': False, # normalize by col medians
            'rmDCoffset': False, # remove dc offset from start time tomain
            'inSR': None, # input sample rate, for running detectors continuously
            'PCEN': True, # Per channel energy normalization
            'fmin': 150} # minimum frequency to include



#%% Create batch loaders and get size of represnetations


# Create the batch loader which will report the size
trainLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                           trainTest = 'train', batch_size=164,  n_classes=7,  
                           minFreq=0)

testLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                           trainTest = 'test', batch_size=164,  n_classes=7,  
                           minFreq=0)

valLoader =  Eco.BatchLoader2(h5_file_validation, 
                           trainTest = 'train', batch_size=164,  n_classes=7,  
                           minFreq=0)

# get the data size, nuber of classes for creating model
trainLoader.specSize
num_classes =7
input_shape = (128, 1281, 1)

#%% Create, compile, and train model


# Create, compile, and train model
model = Eco.create_wider_model(input_shape, num_classes)
model = Eco.compile_model(model)

# It would be great if we could get tensorboard running here
Eco.train_model(model, trainLoader, testLoader, epochs=20)


model.save(modelOutLoc)

# Confusion matrix resluts using Malahat data
confResults = Eco.batch_conf_matrix(loaded_model = model, 
                                    val_batch_loader = valLoader) 

np.savetxt("MalahatConfMat.csv", confResults, delimiter=",")
