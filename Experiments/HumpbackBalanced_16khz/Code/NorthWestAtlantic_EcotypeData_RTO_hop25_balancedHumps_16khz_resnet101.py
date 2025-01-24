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
import EcotypeDefs as Eco
from keras.models import load_model
import librosa
import os

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
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Shuffle the DataFrame rows for debugging
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)


    
#%% Test the parameters by making example spectrograms

# Set up audio parameters
AudioParms = {
            'clipDur': 3,
            'outSR': 16000,
            'nfft': 1024,
            'hop_length':102,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None, 
            'PCEN': True,
            'fmin': 0,
            'min_freq': None,       # default minimum frequency to retain
            'spec_power':2,
            'returnDB':True,         # return spectrogram in linear or convert to db 
            'PCEN_power':31,
            'time_constant':.8,
            'eps':1e-6,
            'gain':0.08,
            'power':.25,
            'bias':10,
            'fmax':16000,
            'Scale Spectrogram': False,
            'AnnotationTrain': trainLoc,
            'AnnotationsTest': testLoc,
            'AnnotationsVal': valLoc,
            'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
            'using batch norm and Raven parameters with Mel Spectrograms and PCEN '} # scale the spectrogram between 0 and 1

# #%% Make the HDF5 files for training testing and evaluation
h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/HumpbackBalanced_16khz/data/Malahat_MnBalanced_16khz_1024fft_PCEN_RTW.h5'
h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/HumpbackBalanced_16khz/Data/MnBalanced_16khz_1024fft_PCEN_RTW.h5'
modelSaveLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced_16khz\\MnBalanced_16khz_1024fft_PCEN_RTW_batchNormResnet50.keras'

# Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, 
#                         parms=AudioParms)


# Eco.create_hdf5_dataset(annotations=annot_val, hdf5_filename= h5_file_validation, 
#                         parms=AudioParms)

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
trainLoader.specSize
num_classes =6
input_shape = (128, 471, 1)

# Create, compile, and train model
model = Eco.create_resnet101(input_shape, num_classes)
model = Eco.compile_model(model)


history = Eco.train_model_history(model, trainLoader, testLoader, epochs=30)
model.save(modelSaveLoc)
metadata = {
    "h5TrainTest": "MnBalanced_16khz_1024fft_PCEN_RTW.h5",
    "h5TrainEval": "Malahat_MnBalanced_16khz_1024fft_PCEN_RTW.h5",
    'ModelFun': 'ResNet50',
    "parameters": {
        "epochs": 30,
        "batch_size": 32,
        "optimizer": "adam"
    }
}

# Save model and metadata



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

# ############################################################################
# # Run model on Eval data
# ###########################################################################
# from keras.models import load_model
# import pandas as pd
# import EcotypeDefs as Eco

# # Load the keras model
# model1 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\MnBalanced_16khz_1024fft_PCEN_RTW_batchNorm.keras')
# annot_test = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/RTO_test.csv")

# # Audio Files to run
# folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\20210728\\'
# folder_path = 'E:\\Malahat\\STN3'


# label_dict = dict(zip(annot_test['label'], annot_test['Labels']))


# # Example detection thresholds (adjust as needed)
# detection_thresholds = {
#     0: 1.5,  # Example threshold for class 0
#     1: 1.5,  # Example threshold for class 1
#     2: 0.25,  # Example threshold for class 2
#     3: 0.25,  # Example threshold for class 3
#     4: 0.25,  # Example threshold for class 4
#     5: 1.5,  # Example threshold for class 5
# }

# class_names = {
#     0: 'AB',
#     1: 'HW',
#     2: 'RKW',
#     3: 'OKW',
#     4: 'TKW',
#     5: 'UndBio'
# }





# fasterProcesser = Eco.AudioProcessor5(
#     folder_path='C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\20210728', 
#     segment_duration=AudioParms['clipDur'], overlap=0, 
#     params=AudioParms, model=model1, class_names=class_names,
#     detection_thresholds=detection_thresholds,
#     selection_table_name="SMRU_20241130_MnBalanced_16khz_1024fft_PCEN_RTW_batchNorm.txt")

# fasterProcesser.process_all_files()

# # Initialize the AudioProcessor with your model and detection thresholds
# processor = Eco.AudioProcessorSimple(folder_path=folder_path,  
#                                      model=model1,
#                            detection_thresholds=detection_thresholds, 
#                            class_names=class_names,
#                            params= AudioParms,
#                            table_type="sound",
#                            overlap=0.25)

# # Process all audio files in the directory
# processor.process_all_files()











