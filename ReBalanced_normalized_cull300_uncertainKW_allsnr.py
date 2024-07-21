# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:04:10 2024

@author: kaity
"""


# This model will
# 1) Use the new balanced train/test set that is balanced across deployments
# and classes
# 2) remove the dc offset in the audio segment before creating the mel spectrogram
# 3) create the HDF5 using the val too for, like, sanity


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model

# updated train/test/ val
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest2.csv")
annot_test = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat2.csv")


AllAnno = pd.concat([annot_train, annot_test], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

# Just using the shortened labels
# Remove the 'labels' column
AllAnno.drop(columns=['label'], inplace=True)
annot_val.drop(columns=['label'], inplace=True)

# Rename 'labelsshort' column to 'labels'
AllAnno.rename(columns={'labelshort': 'label'}, inplace=True)
annot_val.rename(columns={'labelshort': 'label'}, inplace=True)


# Shuffle the DataFrame rows for testing
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)
label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()

# Create the HDF5 files
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_5class_8khz.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_file)

# Create the malahat validation file
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_Malahat_melSpec_5class_8khz.h5'
Eco.create_hdf5_dataset(annotations=annot_val, hdf5_filename= h5_file)


# Calculate frequency of each value in 'var'
val_couts = annot_val['label'].value_counts()

##############################################################################
# Cool now train a model with the rebalanced data
##############################################################################

import EcotypeDefs as Eco



h5_file = 'ReBalanced_melSpec_5class_8khz.h5'

# Create the train and test batch loaders
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=5,    
                              minFreq=300)


test_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=5,
                           minFreq=300)

# Create the Resnet 
num_classes =5
test_batch_loader.specSize
input_shape = (116, 314, 1)

# Create and compile model
smallResnetTanH = Eco.create_model_with_resnet(input_shape, num_classes, 
                                               actName = 'relu')
smallResnetTanH = Eco.compile_model(smallResnetTanH)


# Train model
smallResnetTanH = keras.saving.load_model("C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_5class_8khz_300hz.keras")
Eco.train_model(smallResnetTanH, train_batch_loader, test_batch_loader, epochs=20)
smallResnetTanH.save('C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_5class_8khz_300hz.keras')


# Create the confusion matrix

h5_file = 'ReBalanced_Malahat_melSpec_5class_8khz.h5'
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250,  n_classes=5,
                           minFreq=300)

confResults = Eco.batch_conf_matrix(    loaded_model= smallResnetTanH, 
                                val_batch_loader=val_batch_loader)
############################################################################
# Get the predictions from the hdf5 file
############################################################################
# 2024-07-05
# I've added batch normalization to the create_model_with_resnet function 
# so probably worht rerunning

import EcotypeDefs as Eco



h5_file = 'ReBalanced_melSpec_5class_8khz.h5'

# Create the train and test batch loaders
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=5,
                           minFreq=0)


test_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=5,
                           minFreq=0)

# Create the Resnet 
num_classes =5
test_batch_loader.specSize
input_shape = (128, 314, 1)

# Create and compile model
smallResnetTanH = Eco.create_model_with_resnet(input_shape, num_classes, 
                                               actName = 'relu')
smallResnetTanH = Eco.compile_model(smallResnetTanH)


# Train model

Eco.train_model(smallResnetTanH, train_batch_loader, test_batch_loader, epochs=20)
smallResnetTanH.save('C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_5class_8khz_batchNorm_relu.keras')

# this really isn't working so I killed it. I think there is an issue when using
# batch norm? Restarting just in case.

# Create the confusion matrix
h5_file = 'ReBalanced_Malahat_melSpec_5class_8khz.h5'
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250,  n_classes=5,
                           minFreq=300)

confResults = Eco.batch_conf_matrix(    loaded_model= smallResnetTanH, 
                                val_batch_loader=val_batch_loader)
############################################################################
# Get the predictions from the hdf5 file
############################################################################


import h5py
import pandas as pd
import librosa
import math
import numpy as np
import keras

h5_file = 'ReBalanced_Malahat_melSpec_5class_8khz.h5'
hf = h5py.File(h5_file, 'r')

data_keys = list(hf['train'].keys())
data =hf['train'][data_keys[1]]['spectrogram']

# If a frequency limit is set then  figure out what that is now
minFreq = 300
specSize = data.shape

#This is for trimming the frequency range
mel_frequencies = librosa.core.mel_frequencies(n_mels= specSize[0]+2)

# Find the index corresponding to 500 Hz in mel_frequencies
minIdx = np.argmax(mel_frequencies >= minFreq)
    
#also update the spectrogram size
data= data[minIdx:,:]
specSize = data.shape



# Load the mode
smallResnetTanH = keras.saving.load_model("C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_melSpec_5class_8khz_300hz.keras")


# Step 1: Pre-allocate DataFrame
n = len(hf['train'])  # Number of rows (replace with your desired number)
columns = ['index', 'true',  'AB', 'BKW','NRKW', 'OFFSHORE', 'SRKW', 'SNR']
preds = pd.DataFrame(columns=columns)


rows_list = []

batch_size =1000
iter_len = np.int16(np.ceil(n/batch_size))
indexes = np.arange(n)


combined_dict = {
    'indexs': [],
    'true': [],
    'AB': [],
    'BKW': [],
    'SRKW': [],
    'NRKW': [],
    'OFFSHORE': [],
    'SNR':[]
    
}

# Step 2: Open HDF5 file and iterate through indices
for index in range(iter_len):
    start_index = index * batch_size
    end_index = min((index + 1) * batch_size, n)
    
    batch_data = []
    batch_labels = []
    batch_utc = []
    batch_keys = []
    batch_snr =[]
    
    # Iterate through each batch
    for i in range(start_index, end_index):
        key = data_keys[indexes[i]]
        spec = hf['train'][key]['spectrogram'][minIdx:, :]
        label = hf['train'][key]['label'][()]
        utc = np.array(hf['train'][key]['UTC'])
        snr = np.array(hf['train'][key]['SNR'])
        
        batch_data.append(spec)
        batch_labels.append(label)
        batch_utc.append(utc)
        batch_keys.append(key)
        batch_snr.append(snr)
    
    batch_data = np.array(batch_data)
    prediction = smallResnetTanH.predict(batch_data)
    
    # Append predictions to combined_dict
    combined_dict['indexs'].extend(np.int16(batch_keys))
    combined_dict['true'].extend(batch_labels)
    combined_dict['AB'].extend(prediction[:, 0])
    combined_dict['BKW'].extend(prediction[:, 1])
    combined_dict['SRKW'].extend(prediction[:, 4])
    combined_dict['NRKW'].extend(prediction[:, 2])
    combined_dict['OFFSHORE'].extend(prediction[:, 3])
    combined_dict['SNR'].extend(batch_snr)

# Example usage after the loop completes
# Print or process combined_dict as needed
print(combined_dict)


results =pd.DataFrame.from_dict(combined_dict)
# Find column with maximum value and the maximum value itself
max_column = results[['AB', 'BKW', 'SRKW', 'OFFSHORE', 'NRKW']].idxmax(axis=1)
max_value = results[['AB', 'BKW', 'SRKW', 'OFFSHORE', 'NRKW']].max(axis=1)

# Create new columns in the original DataFrame
results['Pred'] = max_column
results['Pred_value'] = max_value


annot_val['indexs'] = annot_val.index

merged_df = pd.merge(results, annot_val, left_on='indexs', right_on='indexs')
merged_df['Correct'] = merged_df['Pred']== merged_df['Labels']


# do it in r, write to csv

# Specify the path where you want to save the CSV file
csv_file_path = 'NN_scores_ReBalanced_melSpec_5class_8khz_300hz.csv'

# Write DataFrame to CSV file
merged_df.to_csv(csv_file_path, index=False)




# Step 6: Optionally save the DataFrame to a new HDF5 file or use it further
df.to_hdf('output_data.h5', key='df', mode='w')  # Save to HDF5 if needed






To merge two Pandas DataFrames results and annot_val based on the 'indexes' column from results and the index of annot_val, you can use the merge() function in Pandas. Here's how you can do it:

python

import pandas as pd

# Assuming 'results' is your DataFrame with 'indexes' column
# and 'annot_val' is another DataFrame

# Example DataFrames (replace with your actual DataFrames)
resultsa = pd.DataFrame({
    'indexes': ['key1', 'key2', 'key3'],  # Replace with your actual data
    'column1': [1, 2, 3],                 # Replace with your actual data
    'column2': [4, 5, 6]                  # Replace with your actual data
})

annot_vala = pd.DataFrame({
    'columnA': ['A', 'B', 'C'],            # Replace with your actual data
    'columnB': ['X', 'Y', 'Z']             # Replace with your actual data
})

# Merge based on 'indexes' column in results and index in annot_val
merged_df = pd.merge(resultsa, annot_vala, left_on='indexes', right_index=True)










