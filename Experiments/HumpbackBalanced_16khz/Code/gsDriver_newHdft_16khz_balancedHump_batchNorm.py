# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio repreentations. The 
intention is to produce a full pipeline of audio processing

Model used non-normalized spectrograms but batch normalization in the model

"""
# Driver to train neural network on google cloud

import numpy as np
import EcotypeDefs as Eco
import google.cloud.storage
import pickle


# Save history to a pickle file
def save_history_to_pickle(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)



# Define GCS bucket and blob paths
bucket_name = 'dclde2026_hdf5s'


# Set up audio parameters
AudioParms = {
            'clipDur': 3,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':256,
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
            'Scale Spectrogram': False} # scale the spectrogram between 0 and 1
# Create the batch loader instances for streaming data from GCS


# Define local HDF5 file paths
hdf5_file_path_traintest = '/home/kpalmer/Balanced_melSpec_8khz_PCEN_RTW.h5'
df5_file_path_val='/home/kpalmer/Malahat_Balanced_melSpec_8khz_PCEN_RTW.h5'


num_classes =6

# Create the batch loader instances
trainLoader = Eco.BatchLoader2(hdf5_file_path_traintest, 
                               trainTest='train', batch_size=164, n_classes=num_classes, 
                               minFreq=0)

testLoader = Eco.BatchLoader2(hdf5_file_path_traintest, 
                              trainTest='test', batch_size=164, n_classes=num_classes, 
                              minFreq=0)

valLoader = Eco.BatchLoader2(df5_file_path_val, 
                             trainTest='train', batch_size=400, n_classes=num_classes, 
                             minFreq=0)

# Get the data size
trainLoader.specSize
input_shape = (128, 94, 1)


# Create, compile, and train model
# Create, compile, and train model
model = Eco.ResNet18_batchNorm(input_shape, num_classes)
model = Eco.compile_model(model)
history = Eco.train_model_history(model, trainLoader, testLoader, epochs=40)

# Save the model and the history locally
model_save_path = 'MnBalanced_8khz_Resnet18_RTO_batchNorm.keras'
save_history_to_pickle(history, 'training_history_MnBalanced_8khz_Resnet18_RTO_batchNorm.pkl')
model.save(model_save_path)



# Confusion matrix results using Malahat data
# Instantiate the evaluator
label_dict={0: 'AB', 1: 'HW', 4: 'TKW', 5: 'UndBio', 2: 'RKW', 3: 'OKW'}
evaluator = Eco.ModelEvaluator( loaded_model=model, 
                               val_batch_loader = valLoader, 
                               label_dict =label_dict)
# Run the evaluation (only once)
evaluator.evaluate_model()

# Get the various outputs for model checking
confResults, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
scoreDF = evaluator.score_distributions()
pr_curves = evaluator.precision_recall_curves()

np.savetxt("MnBalanced_8khz_Resnet18_RTO_batchnormSpecConfMat.csv", confResults, delimiter=",")

# Upload data to output

###########################################################################
# Upload outputs to GCS
##########################################################################

# Function to upload files to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}.")
    

import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f'MnBalanced_16khz_Resnet18_{timestamp}.keras'
history_save_path = f'training_history_{timestamp}.pkl'
conf_matrix_save_path = f'ConfMat_{timestamp}.csv'


        
try:
    upload_blob(bucket_name, model_save_path, f'output/{model_save_path}')
    upload_blob(bucket_name, history_save_path, f'output/{history_save_path}')
    upload_blob(bucket_name, conf_matrix_save_path, f'output/{conf_matrix_save_path}')
    print("All files uploaded successfully.")
except Exception as e:
    print(f"Error during file upload: {e}")




# # Upload the trained model and confusion matrix to Google Cloud Storage
# output_model_blob_name = 'output/MnBalanced_16khz_Resnet18_RTO_normSpec.keras'
# upload_blob(bucket_name, model_save_path, output_model_blob_name)
# upload_blob(bucket_name, "MnBalanced_16khz_Resnet18_RTO_normSpecConfMat.csv", 
#             'output/MnBalanced_16khz_Resnet18_RTO_normSpecConfMat.csv')
# upload_blob(bucket_name, "training_history_MnBalanced_16khz_Resnet18_RTO_normSpec.json", 
#             'output/training_history_MnBalanced_16khz_Resnet18_RTO_normSpec.json')

