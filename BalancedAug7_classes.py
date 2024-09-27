# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio repreentations. The 
intention is to produce a full pipeline of audio processing

"""


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model


# New more balanced train/test dataset so load that then create the HDF5 database
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest2.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Shuffle the DataFrame rows for testing
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)

label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()

# Create the HDF5 file
# Spectrogram parameters
AudioParms = {
            'clipDur': 2,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':102,
            'spec_type': 'mel',  # Assuming mel spectrogram is used
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None}
    

h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_SNR.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, parms=AudioParms)


# Create the database for the the validation data
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat2.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\MalahatBalanced_melSpec_8khz_SNR.h5'
Eco.create_hdf5_dataset(annotations=h5_file_validation,
                        hdf5_filename= h5_file, parms = AudioParms)


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

# get the data size
trainLoader.specSize
num_classes =7
input_shape = (128, 314, 1)

# Create, compile, and train model
model = Eco.create_model_with_resnet(input_shape, num_classes)
model = Eco.compile_model(model)
Eco.train_model(model, trainLoader, testLoader, epochs=20)
model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_SNR.keras')

########################################################################

confResults = Eco.confuseionMat(model= model, 
                                val_batch_loader=valLoader)

########################################################################


# Now run it with the pipeline



    # Load Keras model
    model_path = 'C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_SNR.keras'
    model = load_model(model_path)

    # Spectrogram parameters
    audio_params = {
        'clipDur': 2,
        'outSR': 16000,
        'nfft': 512,
        'hop_length': 102,
        'spec_type': 'mel',  # Assuming mel spectrogram is used
        'rowNorm': True,
        'colNorm': True,
        'rmDCoffset': True,
        'inSR': None
    }
    
    # Example detection thresholds (adjust as needed)
    detection_thresholds = {
        0: 0.8,  # Example threshold for class 0
        1: 0.8,  # Example threshold for class 1
        2: 0.9,  # Example threshold for class 2
        3: 0.8,  # Example threshold for class 3
        4: 0.8,  # Example threshold for class 4
        5: 0.8,  # Example threshold for class 5
        6: 0.9   # Example threshold for class 6
    }
    
    class_names = {
        0: 'Abiotic',
        1: 'BKW',
        2: 'HW',
        3: 'NRKW',
        4: 'Offshore',
        5: 'SRKW',
        6: 'Und Bio'
    }
    
    # Example usage:
    folder_path = 'C:\\TempData\\Malahat\\STN3\\20151028'
    
    # Initialize the AudioProcessor with your model and detection thresholds
    processor = Eco.AudioProcessor(folder_path=folder_path, model=model,
                               detection_thresholds=detection_thresholds, 
                               class_names=class_names,    
                               params = audio_params,
                               overlap=0.25)
    
    # Process all audio files in the directory
    processor.process_all_files()
























# # Load the hdf5 file and create a histogram of SNR values from the hdf5
# hf = h5py.File(h5_file, 'r')

# # Extract all 'SNR' values from 'train' group using list comprehension
# train_snr_values = np.array([hf['train'][key]['SNR'][()] for key in hf['train']])

# # Close the HDF5 file
# hf.close()

# # Create histogram using Matplotlib
# plt.figure(figsize=(10, 6))
# plt.hist(train_snr_values.flatten(), bins=30, alpha=0.5, label='Train SNR')

# plt.xlabel('SNR Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of SNR Values in Train Group')
# plt.legend()

# plt.grid(True)
# plt.show()



