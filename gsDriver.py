# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio repreentations. The 
intention is to produce a full pipeline of audio processing

"""
# Driver to train neural network on google cloud


import h5py
import pandas as pd
import numpy as npp
import EcotypeDefs as Eco
import librosa


# New more balanced train/test dataset so load that then create the HDF5 database
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2_Edrive.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest2_Edrive.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]


label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()

    

# Exclude any audio files from JASCO that have 'HF' in tehm
# Exclude any data with HPF
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]
AllAnno = AllAnno[~AllAnno['Soundfile'].str.contains('HPF.wav')]


# Set up audio parameters
AudioParms = {
            'clipDur': 2,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':25,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None, 
            'PCEN': True,
            'fmin': 300}

h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_SNR_PCEN_300Hz.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, parms=AudioParms)


# Create the database for the the validation data
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat2.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\MalahatBalanced_melSpec_8khz_PCEN_300hz.h5'
Eco.create_hdf5_dataset(annotations=annot_val,
                        hdf5_filename= h5_file_validation, parms = AudioParms)


#%%


# Exclude any audio files from JASCO that have 'HF' in tehm
# Exclude any data with HPF
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]
AllAnno = AllAnno[~AllAnno['Soundfile'].str.contains('HPF.wav')]


# Set up audio parameters
AudioParms = {
            'clipDur': 2,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':25,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None, 
            'PCEN': True,
            'fmin': 150}

h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_SNR_PCEN.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, parms=AudioParms)


# Create the database for the the validation data
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat2.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\MalahatBalanced_melSpec_8khz_PCEN.h5'
Eco.create_hdf5_dataset(annotations=annot_val,
                        hdf5_filename= h5_file_validation, parms = AudioParms)



import numpy as np
#import matplotlib.pyplot as plt
import EcotypeDefs as Eco
#from keras.models import load_model
#import librosa
import google.cloud.storage

# Define GCS bucket and blob paths
bucket_name = 'dclde2026_hdf5s'


# Set up audio parameters
AudioParms = {
    'clipDur': 2,
    'outSR': 16000,
    'nfft': 512,
    'hop_length': 25,
    'spec_type': 'mel',  
    'spec_power': 2,
    'rowNorm': False,
    'colNorm': False,
    'rmDCoffset': False,
    'inSR': None, 
    'PCEN': True,
    'fmin': 150
}

# Create the batch loader instances for streaming data from GCS

# # Initialize the BatchLoader2 with your settings
# trainLoader = Eco.BatchLoaderGScloud(
#     hdf5_file=hdf5_file_path_traintest,
#     batch_size=164,          # Set your desired batch size
#     trainTest='train',       # Set to 'train' or 'test' based on your needs
#     shuffle=True,            # Shuffle data after every epoch
#     n_classes=7,             # Number of classes in your labels
#     return_data_labels=False, # If True, returns data and labels separately
#     minFreq=0              # Frequency limit (optional, adjust as needed)
# )

# testLoader = Eco.BatchLoaderGScloud(
#     hdf5_file=hdf5_file_path_traintest,
#     batch_size=164,          # Set your desired batch size
#     trainTest='test',       # Set to 'train' or 'test' based on your needs
#     shuffle=True,            # Shuffle data after every epoch
#     n_classes=7,             # Number of classes in your labels
#     return_data_labels=False, # If True, returns data and labels separately
#     minFreq=0              # Frequency limit (optional, adjust as needed)
# )

# valLoader = Eco.BatchLoaderGScloud(
#     hdf5_file=hdf5_file_path_val,
#     batch_size=164,          # Set your desired batch size
#     trainTest='test',       # Set to 'train' or 'test' based on your needs
#     shuffle=True,            # Shuffle data after every epoch
#     n_classes=7,             # Number of classes in your labels
#     return_data_labels=False, # If True, returns data and labels separately
#     minFreq=0              # Frequency limit (optional, adjust as needed)
# )


# Define local HDF5 file paths
df5_file_path_val = '/home/kpalmer/MalahatBalanced_melSpec_8khz_PCEN.h5'
hdf5_file_path_traintest = '/home/kpalmer/Balanced_melSpec_8khz_SNR_PCEN.h5'


# hdf5_file_path_traintest = '/home/kpalmer/Balanced_melSpec_8khz_SNR_PCEN_test.h5'
# hdf5_file_path_val = '/home/kpalmer/MalahatBalanced_melSpec_8khz_PCEN_test.h5'



# Create the batch loader instances
trainLoader = Eco.BatchLoader2(hdf5_file_path_traintest, 
                               trainTest='train', batch_size=164, n_classes=7, 
                               minFreq=0)

testLoader = Eco.BatchLoader2(hdf5_file_path_traintest, 
                              trainTest='test', batch_size=164, n_classes=7, 
                              minFreq=0)

valLoader = Eco.BatchLoader2(df5_file_path_val, 
                             trainTest='train', batch_size=500, n_classes=7, 
                             minFreq=0)

# Get the data size
valLoader.specSize
num_classes = 7
input_shape = (128, 126, 1)
#input_shape = (128, 1281,1)

# Create, compile, and train model
model = Eco.ResNet18(input_shape, num_classes)
model = Eco.compile_model(model)
Eco.train_model(model, trainLoader, testLoader, epochs=20)


# Save the model locally
model_save_path = 'resnet18_PECN_melSpec.keras'
model.save(model_save_path)

# Confusion matrix results using Malahat data
confResults = Eco.batch_conf_matrix(loaded_model=model, val_batch_loader=valLoader) 
np.savetxt("MalahatConfMat.csv", confResults, delimiter=",")

# Function to upload files to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")

# Upload the trained model and confusion matrix to Google Cloud Storage
output_model_blob_name = 'output/resnet18_PECN_melSpec.keras'
upload_blob(bucket_name, model_save_path, output_model_blob_name)
upload_blob(bucket_name, "MalahatConfMat.csv", 'output/MalahatConfMat.csv')
