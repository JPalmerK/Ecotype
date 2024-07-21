# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from EcotypeDefs import create_hdf5_dataset, train_model,create_model_with_resnet
from EcotypeDefs import BatchLoader2, compile_model


# New more balanced train/test dataset so load that then create the HDF5 database

annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()

# Create the HDF5 file
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/TrainTest_08_khz_Melint16MetricsSNR_Balanced.h5'
create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_file)

# Create the Resnet 
num_classes =7
input_shape = (128, 314, 1)

# Create and compile model
model = create_model_with_resnet(input_shape, num_classes)
model = compile_model(model)

# Create the train and test batch loaders
train_batch_loader = BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=7)
test_batch_loader =  BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=7)

# Train model
train_model(model, train_batch_loader, test_batch_loader, epochs=12)













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



