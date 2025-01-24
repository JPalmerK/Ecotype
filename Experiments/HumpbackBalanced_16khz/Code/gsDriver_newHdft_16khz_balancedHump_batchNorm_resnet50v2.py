# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio representations.
The intention is to produce a full pipeline of audio processing.

Model used non-normalized spectrograms but batch normalization in the model.
"""
import numpy as np
import json
import datetime
import os
import google.cloud.storage
import EcotypeDefs as Eco

# Define GCS bucket and blob paths
bucket_name = 'dclde2026_hdf5s'

# Set up audio parameters
trainLoc = "C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced\\Data\\RTO_train_HumpBalanced.csv"
testLoc = "C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced\\Data\\RTO_test_HumpBalanced.csv"
valLoc = 'C:/Users/kaity/Documents/GitHub/Ecotype/RTO_malahat.csv'

AudioParms = {
    'clipDur': 3,
    'outSR': 16000,
    'nfft': 1024,
    'hop_length': 102,
    'spec_type': 'mel',
    'spec_power': 2,
    'rowNorm': False,
    'colNorm': False,
    'rmDCoffset': False,
    'inSR': None,
    'PCEN': True,
    'fmin': 0,
    'min_freq': None,
    'returnDB': True,
    'PCEN_power': 31,
    'time_constant': 0.8,
    'eps': 1e-6,
    'gain': 0.08,
    'power': 0.25,
    'bias': 10,
    'fmax': 16000,
    'Scale Spectrogram': False,
    'AnnotationTrain': trainLoc,
    'AnnotationsTest': testLoc,
    'AnnotationsVal': valLoc,
    'Notes': 'Balanced humpbacks by removing a bunch of humpbacks randomly, '
             'using batch norm and Raven parameters with Mel Spectrograms and PCEN.'
}

# Define local HDF5 file paths
# Define base filenames
base_filename_traintest = "MnBalanced_16khz_1024fft_PCEN_RTW.h5"
base_filename_eval = "Malahat_MnBalanced_16khz_1024fft_PCEN_RTW.h5"


# Define base directory for file paths
base_dir = "/home/kpalmer/"

# Construct full paths dynamically
trainTestFname = base_filename_traintest
evalFNAME = base_filename_eval
hdf5_file_path_traintest = f"{base_dir}{base_filename_traintest}"
hdf5_file_path_val = f"{base_dir}{base_filename_eval}"


num_classes = 6
input_shape = (128, 471, 1)
batch_size =64
maxEpocs =40

# Create the batch loader instances
trainLoader = Eco.BatchLoader2(hdf5_file_path_traintest, trainTest='train', 
                               batch_size=batch_size, n_classes=num_classes, minFreq=0)
testLoader = Eco.BatchLoader2(hdf5_file_path_traintest, trainTest='test',
                              batch_size=batch_size, n_classes=num_classes, minFreq=0)
valLoader = Eco.BatchLoader2(hdf5_file_path_val, trainTest='train', 
                             batch_size=400, n_classes=num_classes, minFreq=0)

# Create, compile, and train the model
model = Eco.create_resnet50(input_shape, num_classes)
#model = Eco.ResNet1_testing(input_shape, num_classes)
model = Eco.compile_model(model)
history = Eco.train_model_history(model, trainLoader, testLoader, epochs=maxEpocs)

# Save the model locally
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f'MnBalanced_16khz_Resnet50_{timestamp}.keras'
model.save(model_save_path)
print(f"Model saved at {model_save_path}.")

# Save metadata to JSON
import json
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metadata = {
    'h5TrainTest': 'MnBalanced_16khz_1024fft_PCEN_RTW.h5',
    'h5TrainEval': 'Malahat_MnBalanced_16khz_1024fft_PCEN_RTW.h5',
    'ModelFun': 'ResNet50',
    'parameters': {
        'epochs': 1,
        'batch_size': 64,
        'optimizer': 'adam',
        'input_shape': (128, 471, 1),
        'num_classes': 6,
        'audio_params': {
            'clipDur': 3,
            'outSR': 16000,
            'nfft': 1024,
            'hop_length': 102,
            'spec_type': 'mel',
            'spec_power': 2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None,
            'PCEN': True,
            'fmin': 0,
            'min_freq': None,
            'returnDB': True,
            'PCEN_power': 31,
            'time_constant': 0.8,
            'eps': 1e-06,
            'gain': 0.08,
            'power': 0.25,
            'bias': 10,
            'fmax': 16000,
            'Scale Spectrogram': False,
            'AnnotationTrain': 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced\\Data\\RTO_train_HumpBalanced.csv',
            'AnnotationsTest': 'C:/Users/kaity/Documents/GitHub/Ecotype/Experiments\\HumpbackBalanced\\Data\\RTO_test_HumpBalanced.csv',
            'AnnotationsVal': 'C:/Users/kaity/Documents/GitHub/Ecotype/RTO_malahat.csv',
            'Notes': 'Balanced humpbacks by removing a bunch of humpbacks randomly, using batch norm and Raven parameters with Mel Spectrograms and PCEN.'
        }
    },
    'Notes': 'Balanced humpbacks by removing a bunch of humpbacks randomly, using batch norm and Raven parameters with Mel Spectrograms and PCEN.'
}

metadata_save_path = f"metadata_{timestamp}_resnet50.json"
print(metadata_save_path)  # Confirm save path for debugging

# Save metadata to a file
with open(metadata_save_path, "w") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

# Ensure this print statement is outside the 'with' block.
print(f"Metadata saved at {metadata_save_path}.")



# Save training history to JSON
history_dict = history.history  # Extract history dictionary
history_save_path = f"training_history_resnet50{timestamp}.json"
with open(history_save_path, "w") as history_file:
    json.dump(history_dict, history_file, indent=4)
print(f"Training history saved at {history_save_path}.")

# Function to upload files to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")

# Upload the model, metadata, and history to GCS
try:
    upload_blob(bucket_name, model_save_path, f'output/{model_save_path}')
    upload_blob(bucket_name, metadata_save_path, f'output/{metadata_save_path}')
    upload_blob(bucket_name, history_save_path, f'output/{history_save_path}')
    print("All files uploaded successfully.")
except Exception as e:
    print(f"Error during file upload: {e}")

# Confusion matrix results using validation data
label_dict = {1: 'HW', 2: 'RKW', 0: 'AB', 4: 'TKW', 5: 'UndBio', 3: 'OKW'}
evaluator = Eco.ModelEvaluator(loaded_model=model, 
                               val_batch_loader=valLoader, 
                               label_dict=label_dict)

confResults, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
conf_matrix_save_path = f'ConfMat_resnet50{timestamp}.csv'
np.savetxt(conf_matrix_save_path, confResults, delimiter=",")
print(f"Confusion matrix saved at {conf_matrix_save_path}.")

# Upload the confusion matrix to GCS
try:
    upload_blob(bucket_name, conf_matrix_save_path, f'output/{conf_matrix_save_path}')
    print("Confusion matrix uploaded successfully.")
except Exception as e:
    print(f"Error uploading confusion matrix: {e}")
