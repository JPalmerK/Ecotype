# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio repreentations. The 
intention is to produce a full pipeline of audio processing

"""
# Driver to train neural network on google cloud

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
model = Eco.create_wider_model2(input_shape, num_classes)
model = Eco.compile_model(model)
Eco.train_model(model, trainLoader, testLoader, epochs=20)


# Save the model locally
model_save_path = 'resnet12_PECN_melSpec.keras'
model.save(model_save_path)

# Confusion matrix results using Malahat data
confResults = Eco.batch_conf_matrix(loaded_model=model, val_batch_loader=valLoader) 
np.savetxt("MalahatConfMat_wider_model2.csv", confResults, delimiter=",")

# Function to upload files to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")

# Upload the trained model and confusion matrix to Google Cloud Storage
output_model_blob_name = 'output/resnet12_PECN_melSpec.keras'
upload_blob(bucket_name, model_save_path, output_model_blob_name)
upload_blob(bucket_name, "MalahatConfMat_wider_model2.csv", 'output/MalahatConfMat_wider_model2.csv')
