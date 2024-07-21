# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:38:51 2024

@author: kaity
"""

import h5py
import pandas as pd 
import numpy as np
import EcotypeDefs as Eco.

# updated train/test/ val
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat2.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

# Just using the shortened labels
# Remove the 'labels' column
annot_val.drop(columns=['label'], inplace=True)

# Rename 'labelsshort' column to 'labels'
annot_val.rename(columns={'labelshort': 'label'}, inplace=True)


hdf5_filename = 'C:/Users/kaity/Documents/GitHub/Ecotype/ReBalanced_Malahat_melSpec_5class_8khz.h5'
hf = h5py.File(hdf5_filename, 'w')

h5keys  = list(hf['train'].keys())


    
for ii in range(0, len(annot_val)):    
    row = annot_val.iloc[ii]
    utc = row['UTC']
    
    # get the dataset
    dataset = hf['train'][str(ii)]
    dataset.create_dataset(f'{ii}/UTC', data=utc)
    print(ii, ' of ', len(annot_val))
    
    
    
    