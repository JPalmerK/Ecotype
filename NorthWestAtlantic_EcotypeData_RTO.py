# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio repreentations. The 
intention is to produce a full pipeline of audio processing


Training data included all datasets
All frequencies
KW labels for resident tranisent and offshore only all other labels


"""





import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model
import librosa
import os

np.random.seed(seed=2)

# New more balanced train/test dataset so load that then create the HDF5 database
# annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2_Edrive.csv")
# annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest2_Edrive.csv")

# New more balanced train/test dataset so load that then create the HDF5 database
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/RTO_train.csv")
annot_test = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/RTO_test.csv")
annot_val = pd.read_csv('C:/Users/kaity/Documents/GitHub/Ecotype/RTO_malahat.csv')


AllAnno = pd.concat([annot_train, annot_test], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Shuffle the DataFrame rows for debugging
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)


# Function to create and save a spectrogram plot
def makePlot(spectrogram, save_path):
    if isinstance(spectrogram, np.ndarray):
        spectrogram = spectrogram.astype(np.float32)
    else:
        raise TypeError("Spectrogram should be a NumPy array.")
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
    plt.imshow(np.array(spec), 
                aspect='auto', 
                origin='lower', 
                cmap='viridis')  # Adjust parameters as needed
    plt.colorbar(label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path, format='png')  # Save the figure
    plt.close()  # Close the plot to free up memory

    
    
#%% Test the parameters by making example spectrograms


# Set up audio parameters
AudioParms = {
            'clipDur': 3,
            'outSR': 8000,
            'nfft': 512,
            'hop_length':256,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None, 
            'PCEN': True,
            'fmin': 0}




#%% Make the HDF5 files for training testing and evaluation


h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_PCEN_RTW.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, 
                        parms=AudioParms)


# Create the database for the the validation data
annot_val['traintest'] = 'Train'
h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Malahat_Balanced_melSpec_8khz_PCEN_RTW.h5'
Eco.create_hdf5_dataset(annotations=annot_val,
                        hdf5_filename= h5_file_validation, 
                        parms = AudioParms)


#%%


# Create the batch loader which will report the size
trainLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                           trainTest = 'train', batch_size=164,  n_classes=6,  
                           minFreq=0)

testLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                           trainTest = 'test', batch_size=164,  n_classes=6,  
                           minFreq=0)

valLoader =  Eco.BatchLoader2(h5_file_validation, 
                           trainTest = 'train', batch_size=164,  n_classes=6,  
                           minFreq=0,   return_data_labels = False)

# get the data size
trainLoader.specSize
num_classes =6
input_shape = (128, 94, 1)
#%% Check generators

# for batch in trainLoader:
#     print(f"Batch: {batch}")  # Inspect what `batch` contains
#     if batch is None:
#         print("Found None in trainLoader")
#     else:
#         # Check the type of `batch` and its length
#         print(f"Type of batch: {type(batch)}")
#         print(f"Length of batch: {len(batch)}")

#         # Ensure it is unpackable into `x` and `y`
#         if isinstance(batch, tuple) and len(batch) == 2:
#             x, y = batch
#             if x is None or y is None:
#                 print("Found None in batch")
#         else:
#             print("Batch format is incorrect")



#%% Train the detector



# Create, compile, and train model
model = Eco.ResNet18(input_shape, num_classes)
model = Eco.compile_model(model)
Eco.train_model(model, trainLoader, testLoader, epochs=10)
model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_8khz_RTO.keras')
#%%


label_dict = dict(zip(AllAnno['label'], AllAnno['Labels']))

h5_file_validation1 = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Malahat_Balanced_melSpec_8khz_PCEN_RTW.h5'
model1 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_8khz_RTO.keras')

valLoader1 =  Eco.BatchLoader2(h5_file_validation, 
                           trainTest = 'train', batch_size=164,  n_classes=6,  
                           minFreq=0,   return_data_labels = False)


# Instantiate the evaluator
evaluator = Eco.ModelEvaluator( loaded_model=model1, val_batch_loader = valLoader1, label_dict =label_dict)

# Run the evaluation (only once)
evaluator.evaluate_model()
conf_matrix_df, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
scoreDF = evaluator.score_distributions()




