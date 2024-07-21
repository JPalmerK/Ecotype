# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:38:52 2024

@author: kaity
"""



# This model will
# 1) Use the new balanced train/test set that is balanced across deployments
# and classes
# 2) remove the dc offset in the audio segment and create a spectrogram rather
# than mel spectrogram
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
