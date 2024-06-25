# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:23:32 2024

@author: kaity
"""

import h5py
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load original HDF5 file
original_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.h5'
hf_original = h5py.File(original_file, 'r')

# Step 2: Access datasets and filter SNR values where 'label' == 2
snr_values = []
for group_name in ['train', 'test']:  # Assuming 'train' and 'test' groups
    group = hf_original[group_name]
    for dataset_name, dataset in group.items():
        if 'label' in dataset and 'SNR' in dataset:  # Check if 'label' and 'SNR' exist
            labels = dataset['label']
            snr = dataset['SNR']
            
            # Check if datasets are scalar or array-like
            if isinstance(labels, h5py.Dataset) and isinstance(snr, h5py.Dataset):
                # Read dataset values
                labels_value = labels[()]
                snr_value = snr[()]
                
                # Handle scalar case for 'label'
                if np.isscalar(labels_value):
                    labels_value = np.array([labels_value])
                
                # Handle scalar case for 'SNR'
                if np.isscalar(snr_value):
                    snr_value = np.array([snr_value])
                
                # Filter SNR values where label is 2
                snr_hw = snr_value[labels_value == 2]
                snr_values.extend(snr_hw.tolist())  # Convert to list for histogram plotting

# Step 3: Plot histogram of SNR values where 'label' is 2
plt.figure(figsize=(10, 6))
plt.hist(snr_values, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('SNR Values')
plt.ylabel('Frequency')
plt.title('Histogram of SNR Values for HW Detections')
plt.grid(True)
plt.show()

# Close the HDF5 file
hf_original.close()

import h5py
import numpy as np


# Step 1: Load original HDF5 file
original_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz.h5'
subset_file=  'Balanced_melSpec_8khz_HWcull.h5'



hf_original = h5py.File(original_file, 'r')
subset_file= h5py.File(subset_file, 'w') 


    


def create_subset(original_file, subset_file):
    # Open the original HDF5 file in read mode
    with h5py.File(original_file, 'r') as hf_original:
        
        # Create a new HDF5 file or open an existing one in write mode
        with h5py.File(subset_file, 'w') as hf_subset:
            
            # Process both train and test groups
            for group_name in ['train', 'test']:
                
                # Create a new group in the subset file for the current group
                subset_group = hf_subset.create_group(group_name)
                
                # Iterate through each dataset in the original group
                for key in hf_original[group_name]:
                    original_dataset = hf_original[group_name][key]
                    
                    # Read necessary attributes (assuming they are datasets)
                    label = original_dataset['label'][()]  # Assuming 'label' is a dataset
                    SNR = original_dataset['SNR'][()]      # Assuming 'SNR' is a dataset
                    
                    # Check conditions to decide whether to include this dataset in the subset
                    if label != 2 or SNR >= 7:
                        # Create a corresponding group in the subset file
                        subset_group_key = subset_group.create_group(key)
                        
                        # Copy all attributes from the original dataset to the subset dataset
                        for attribute_name in original_dataset:
                            original_attribute = original_dataset[attribute_name]
                            subset_group_key.create_dataset(attribute_name, data=original_attribute[()])
                            
                        # Optionally, copy additional datasets or attributes as needed
                        # subset_group_key.create_dataset('additional_data', data=some_data)

# Example usage

# Example usage
create_subset(original_file, 'Balanced_melSpec_8khz_HWcull.h5')
##############################################################################

# Cool now train a model
##############################################################################

import EcotypeDefs as Eco
# Create the Resnet 
num_classes =7
input_shape = (128, 314, 1)

# Create and compile model
model = Eco.create_model_with_resnet(input_shape, num_classes)
model = Eco.compile_model(model)


h5_file = 'Balanced_melSpec_8khz_HWcull.h5'

# Create the train and test batch loaders
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=7)
test_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=7)




# Train model
Eco.train_model(model, train_batch_loader, test_batch_loader, epochs=20)
model.save('Balanced_melSpec_8khz_HWcull.keras')

# Evaluate
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatBalanced_melSpec_8khz1.h5'


# Create the confusion matrix
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=1000,  n_classes=7)
confResults = Eco.confuseionMat(model= model, 
                                val_batch_loader=val_batch_loader)

