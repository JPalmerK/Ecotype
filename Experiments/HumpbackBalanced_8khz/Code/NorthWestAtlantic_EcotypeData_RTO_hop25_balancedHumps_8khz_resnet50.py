# # -*- coding: utf-8 -*-
# """
# Created on Fri Jun 21 10:14:13 2024

# @author: kaity

# This is a trial run of code using the separated audio repreentations. The 
# intention is to produce a full pipeline of audio processing


# Training data included all datasets
# All frequencies
# KW labels for resident tranisent and offshore only all other labels


# """

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
import librosa
import os

import sys
sys.path.append("C:/Users/kaity/Documents/GitHub/Ecotype")
import EcotypeDefs as Eco

# Save metadata to JSON
import json
from datetime import datetime


# np.random.seed(seed=2)

# # New more balanced train/test dataset so load that then create the HDF5 database
trainLoc ="C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced\\Data\\RTO_train_HumpBalanced.csv"
testLoc = "C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced\\Data\\RTO_test_HumpBalanced.csv"
valLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/RTO_malahat.csv'



annot_train = pd.read_csv(trainLoc)
annot_test = pd.read_csv(testLoc)
annot_val = pd.read_csv(valLoc)
annot_val['traintest'] = 'Train'

AllAnno = pd.concat([annot_train, annot_test], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 4000]

# Shuffle the DataFrame rows for debugging
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop the field deployments for UAF data, they seem to be corrupt
AllAnno = AllAnno[AllAnno['Dep'] != 'Field_HTI']
AllAnno = AllAnno[AllAnno['Dep'] != 'Field_SondTrap']
    
#%% Test the parameters by making example spectrograms

# Set up audio parameters
AudioParms = {
            'clipDur': 3,
            'outSR': 8000,
            'nfft': 1024,
            'hop_length':102,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': True,
            'inSR': None, 
            'PCEN': True,
            'fmin': 10,
            'min_freq': None,       # default minimum frequency to retain
            'spec_power':2,
            'returnDB':True,         # return spectrogram in linear or convert to db 
            'PCEN_power':31,
            'time_constant':.8,
            'eps':1e-6,
            'gain':0.08,
            'power':.25,
            'bias':10,
            'fmax':8000,
            'Scale Spectrogram': False,
            'AnnotationTrain': trainLoc,
            'AnnotationsTest': testLoc,
            'AnnotationsVal': valLoc,
            'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
            'Excluding UAF data'} # scale the spectrogram between 0 and 1

# #%% Make the HDF5 files for training testing and evaluation
h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/HumpbackBalanced_8khz/Data/Malahat_MnBalanced_8khz_10fmin_1024fft_PCEN_RTW.h5'
h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/HumpbackBalanced_8khz/Data/MnBalanced_8khz_10fmin_1024fft_PCEN_RTW.h5'
modelSaveLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced_8khz\\Output\\Resnet50\\MnBalanced_8khz__10fmin_1024fft_PCEN_RTW_batchNormResnet50.keras'
metadata_save_path ='C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced_8khz\\Output\\Resnet50\\'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, 
                        parms=AudioParms)


Eco.create_hdf5_dataset(annotations=annot_val, hdf5_filename= h5_file_validation, 
                        parms=AudioParms)

# ##############################################################################
# # Train Model
# #############################################################################



#%% Train the detector


# Create the batch loader which will report the size
trainLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                            trainTest = 'train', batch_size=32,  n_classes=6,  
                            minFreq=0)

testLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                            trainTest = 'test', batch_size=32,  n_classes=6,  
                            minFreq=0)

valLoader =  Eco.BatchLoader2(h5_file_validation, 
                            trainTest = 'train', batch_size=32,  n_classes=6,  
                            minFreq=0,   return_data_labels = False)

# get the data size
valLoader.specSize
num_classes =6
input_shape = (128, 236, 1)

# Create, compile, and train model
model = Eco.create_resnet101(input_shape, num_classes)
model = Eco.compile_model(model)


history = Eco.train_model_history(model, trainLoader, testLoader, epochs=30)
model.save(modelSaveLoc)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metadata = {
    'h5TrainTest': 'MnBalanced_16khz_1024fft_PCEN_RTW.h5',
    'h5TrainEval': 'Malahat_MnBalanced_16khz_1024fft_PCEN_RTW.h5',
    'ModelFun': 'ResNet101',
    'parameters': {
        'epochs': 1,
        'batch_size': 64,
        'optimizer': 'adam',
        'input_shape': (128, 471, 1),
        'num_classes': 6,
        'audio_params': {            
            'clipDur': 3,
                    'outSR': 8000,
                    'nfft': 1024,
                    'hop_length':102,
                    'spec_type': 'mel',  
                    'spec_power':2,
                    'rowNorm': False,
                    'colNorm': False,
                    'rmDCoffset': True,
                    'inSR': None, 
                    'PCEN': True,
                    'fmin': 10,
                    'min_freq': None,       # default minimum frequency to retain
                    'spec_power':2,
                    'returnDB':True,         # return spectrogram in linear or convert to db 
                    'PCEN_power':31,
                    'time_constant':.8,
                    'eps':1e-6,
                    'gain':0.08,
                    'power':.25,
                    'bias':10,
                    'fmax':8000,
                    'Scale Spectrogram': False,
                    'AnnotationTrain': trainLoc,
                    'AnnotationsTest': testLoc,
                    'AnnotationsVal': valLoc,
                    'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
                    'Excluding UAF data' }
    },
    'Notes': 'Balanced humpbacks by removing a bunch of humpbacks randomly, using batch norm and Raven parameters with Mel Spectrograms and PCEN.'
}
# Save model and metadata


# Save training history to JSON

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_dict = history.history  # Extract history dictionary
history_save_path = f"training_history_resnet50{timestamp}.json"
with open(history_save_path, "w") as history_file:
    json.dump(history_dict, history_file, indent=4)
print(f"Training history saved at {history_save_path}.")



# Ensure this print statement is outside the 'with' block.
print(f"Metadata saved at {metadata_save_path}.")

#%%


label_dict = dict(zip(AllAnno['label'], AllAnno['Labels']))


model1 = load_model(modelSaveLoc)
# Giggle test shape
model1.input_shape

# Create the validation batch loader and ensure matching parameters
valLoader1 =  Eco.BatchLoader2(h5_file_validation, 
                            trainTest = 'train', batch_size=500,  n_classes=6,  
                            return_data_labels = True)

# Giggle test the parameters
parms = dict(valLoader1.hf.attrs)


# Instantiate the evaluator
evaluator = Eco.ModelEvaluator( loaded_model=model1, 
                               val_batch_loader = valLoader1, 
                               label_dict =label_dict)

# Run the evaluation (only once)
evaluator.evaluate_model()

# Get the various outputs for model checking
conf_matrix_df, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
scoreDF = evaluator.score_distributions()
pr_curves = evaluator.precision_recall_curves()











