# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model


# New more balanced train/test dataset so load that then create the HDF5 database
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Shuffle the DataFrame rows for testing
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)

label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()

# Create the HDF5 file
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_file)
#
# Assuming annotations is your DataFrame containing audio segment metadata
#Eco.create_hdf5_dataset_parallel(AllAnno, 'data_parallel2.h5',  8)


hf = h5py.File(h5_file, 'r')

# Create the Resnet 
num_classes =7
input_shape = (128, 314, 1)

# Create and compile model
model = Eco.create_model_with_resnet(input_shape, num_classes)
model = Eco.compile_model(model)

# Create the train and test batch loaders
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=7)
test_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=7)




# Train model
loaded_model = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.keras')
Eco.train_model(loaded_model, train_batch_loader, test_batch_loader, epochs=18)
loaded_model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.keras')


# Create the database for the the validation data
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatBalanced_melSpec_8khz1.h5'
Eco.create_hdf5_dataset(annotations=annot_val,
                        hdf5_filename= h5_file)

# Create the confusion matrix
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250,  n_classes=7)
confResults = Eco.confuseionMat(model= loaded_model, 
                                val_batch_loader=val_batch_loader)











# Load the hdf5 file and create a histogram of SNR values from the hdf5
hf = h5py.File(h5_file, 'r')

# Extract all 'SNR' values from 'train' group using list comprehension
train_snr_values = np.array([hf['train'][key]['SNR'][()] for key in hf['train']])

# Close the HDF5 file
hf.close()

# Create histogram using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(train_snr_values.flatten(), bins=30, alpha=0.5, label='Train SNR')

plt.xlabel('SNR Values')
plt.ylabel('Frequency')
plt.title('Histogram of SNR Values in Train Group')
plt.legend()

plt.grid(True)
plt.show()



