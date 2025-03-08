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
trainLoc ="C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\MediumDatacetONCfixed_15khz\\Data\\SmallerDataONC_fix_Train_4k.csv"
testLoc = "C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\MediumDatacetONCfixed_15khz\\Data\\SmallerDataONC_fix_Test_4k.csv"
valLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\SmallerSetONCfixed_15khz\\Data\\SmallerDataset_eval_DataONC_fix_Train.csv'

annot_train = pd.read_csv(trainLoc)
annot_test = pd.read_csv(testLoc)
annot_val = pd.read_csv(valLoc)
annot_val['traintest'] = 'Train'

AllAnno = pd.concat([annot_train, annot_test], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Shuffle the DataFrame rows for debugging
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)

# Oops
AllAnno.rename(columns={"Lables": "Labels"}, inplace = True)
annot_val.rename(columns={"Lables": "Labels"}, inplace = True)


# Create a label dictionary to double check everyting is as expected
label_dict =     dict(zip(AllAnno['label'], AllAnno['Labels']))
label_dict_val = dict(zip(annot_val['label'], annot_val['Labels']))


# Step 1: Invert the label_dict to map class names to correct indices
class_to_index = {v: k for k, v in label_dict.items()}

# Step 2: Remap validation labels using the correct indices
corrected_label_dict_val = {k: class_to_index[v] for k, v in label_dict_val.items()}

annot_val['label'] = annot_val['label'].map(corrected_label_dict_val)


#Gaaaah missing 4

sorted_labels = sorted(label_dict.keys())


# Create a mapping to new continuous labels (0 to 6)
new_label_mapping = {old: new for new, old in enumerate(sorted_labels)}

# Apply the mapping to fix label_dict
new_label_dict = {new_label_mapping[k]: v for k, v in label_dict.items()}


print("New label mapping:", new_label_mapping)
print("Updated label_dict:", new_label_dict)

# Apply the mapping to both train/test and validation dataframes
annot_val["label"] = annot_val["label"].map(new_label_mapping)
AllAnno["label"] = AllAnno["label"].map(new_label_mapping)


# Create a label dictionary to double check everyting is as expected
label_dict = dict(zip(AllAnno['label'], AllAnno['Labels']))
dict(zip(annot_val['label'], annot_val['Labels']))

    
#%% Test the parameters by making example spectrograms

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
            'AnnotationTrain': trainLoc,
            'AnnotationsTest': testLoc,
            'AnnotationsVal': valLoc,
            'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
            'using batch norm and Raven parameters with Mel Spectrograms and PCEN '} # scale the spectrogram between 0 and 1

# #%% Make the HDF5 files for training testing and evaluation
h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/MediumDatacetONCfixed_15khz\\data\\Malahat_MnBalanced_15khz_512fft_PCEN_4k_round.h5'
h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments/MediumDatacetONCfixed_15khz/data/MnBalanced_15khz_512fft_PCEN_4k_round.h5'
modelSaveLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\MediumDatacetONCfixed_15khz\\MnBalanced_15khz_512fft_PCEN_4k_resnet101.keras'


Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, 
                        parms=AudioParms)


Eco.create_hdf5_dataset(annotations=annot_val, 
                        hdf5_filename= h5_file_validation, 
                        parms=AudioParms)


# Create the batch loader which will report the size
trainLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                            trainTest = 'train', batch_size=32,  n_classes=7,  
                            minFreq=0)

testLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                            trainTest = 'test', batch_size=32,  n_classes=7,  
                            minFreq=0)

valLoader =  Eco.BatchLoader2(h5_file_validation, 
                            trainTest = 'train', batch_size=200,  n_classes=7,  
                            minFreq=0,   return_data_labels = False)




# get the data size
valLoader.specSize

num_classes =7
input_shape = (128, 883, 1)

#%% Train the detector
# Create, compile, and train model
model = Eco.create_resnet101(input_shape, num_classes)
model = Eco.compile_model(model)


history = Eco.train_model_history(model, trainLoader, testLoader, epochs=30)
model.save(modelSaveLoc)
metadata = {
    "h5TrainTest": "MnBalanced_15khz_512fft_PCEN_RTW.h5",
    "h5TrainEval": "MnBalanced_15khz_512fft_PCEN_RTW_batchNormResnet50.h5",
    'ModelFun': 'ResNet50',
    "parameters": {
                'clipDur': 3,
                'outSR': 15000,
                'nfft': 512,
                'hop_length':51,
                'spec_type': 'mel',  
                'spec_power':0.5,
                'rowNorm': False,
                'colNorm': False,
                'rmDCoffset': False,
                'inSR': None, 
                'PCEN': True,
                'fmin': 0,
                'min_freq': None,       # default minimum frequency to retain
                'returnDB':False,         # return spectrogram in linear or convert to db 
                'NormalizeAudio': True,
                'Scale Spectrogram': True,
                'AnnotationTrain': trainLoc,
                'AnnotationsTest': testLoc,
                'AnnotationsVal': valLoc,
                'Notes' : 'Balanced humpbacks by removing a bunch of humpbacks randomly'+
                'using batch norm and Raven parameters with Mel Spectrograms and PCEN '} }

# Save model and metadata
# import json
# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)


#%%


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
label_dict =  {0: 'NRKW', 
               1: 'OKW', 
               2: 'SRKW', 
               3: 'TKW', 
               4: 'HW',
               6: 'UndBio', 
               5: 'AB'}

evaluator = Eco.ModelEvaluator( loaded_model=model, 
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











