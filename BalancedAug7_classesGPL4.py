# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity

This is a trial run of code using the separated audio repreentations. The 
intention is to produce a full pipeline of audio processing

"""


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model
import librosa


# New more balanced train/test dataset so load that then create the HDF5 database
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2_Edrive.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest2_Edrive.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Shuffle the DataFrame rows for testing
#AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)

label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()


#%%
# #############################################################################
# # Do some testing

import matplotlib.pyplot as plt
import os
import scipy.signal




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


# # Pull out an example of each of each deploymnet and make the figures
# dfSub = df_sorted.drop_duplicates(subset = 'Dep')

# output_dir = 'C:\\Users\\kaity\\Desktop\\TestSpectrograms'
# #output_dir = 'C:\\Users\\kaity\\Desktop\\Basic'
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# AudioParms = {
#             'clipDur': 2,
#             'outSR': 16000,
#             'nfft': 512,
#             'hop_length':25,
#             'spec_type': 'mel',  
#             'spec_power':2,
#             'rowNorm': False,
#             'colNorm': False,
#             'rmDCoffset': False,
#             'inSR': None, 
#             'PCEN': True,
#             'fmin': 150}



# for idx, row in dfSub.iterrows():
#     file_path = row['FilePath']
#     start_time = row['FileBeginSec']
#     end_time = row['FileEndSec']
    
#     # Get the audio data
#     audio_data, sample_rate = librosa.load(file_path, sr=AudioParms['outSR'], 
#                                            offset=start_time,
#                                            duration=AudioParms['clipDur'])
#     # Create the spectrogram
#     spec = Eco.create_spectrogram(audio_data, return_snr=False, **AudioParms)
    
#     # # spec = PCEN(spec)
#     # melspec = librosa.feature.melspectrogram(y=audio_data, 
#     #                                               sr=AudioParms['outSR'], 
#     #                                               n_fft=AudioParms['nfft'], 
#     #                                               hop_length=AudioParms['hop_length'],
#     #                                               fmin =150,
#     #                                               power =2)
    
    
#     # spec =librosa.pcen(
#     #     melspec * (2 ** 31),
#     #     time_constant=0.8,
#     #     eps=1e-6,
#     #     gain=.08,
#     #     power=0.25,
#     #     bias=10,
#     #     sr=AudioParms['outSR'],
#     #     hop_length=AudioParms['hop_length'],
#     # )

#     # Generate the file name for the plot

#     base_name = os.path.splitext(os.path.basename(row['Soundfile']))[0]  # Remove extension
#     file_name = f'spectrogram_{base_name}.png'  # Or .jpg if you prefer
#     save_path = os.path.join(output_dir,file_name)
    
#     # Make and save the plot
#     makePlot(spec, save_path)

# print("Spectrogram plots saved successfully.")
    
    
#%%  cREATE THE HDF5 dabases
#############################################################################

# Send list of hpf5 files onto April

# Filter the DataFrame to get rows where 'Soundfile' contains 'hpf'
hpf_files = AllAnno[AllAnno['Soundfile'].str.contains('hpf', case=False, na=False)]

# Extract the 'Soundfile' column to a list
hpf_file_list = hpf_files['Soundfile'].tolist()

# Create a new DataFrame for the list
hpf_file_df = pd.DataFrame(hpf_file_list, columns=['Soundfile'])

# Save the DataFrame to a CSV file
output_file = 'JASCO_hpf_files.csv'
hpf_file_df.to_csv(output_file, index=False)

#%%


# Exclude any audio files from JASCO that have 'HF' in tehm
# Exclude any data with HPF
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]
AllAnno = AllAnno[~AllAnno['Soundfile'].str.contains('HPF.wav')]


# Set up audio parameters
AudioParms = {
            'clipDur': 2,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':25,
            'spec_type': 'mel',  
            'spec_power':2,
            'rowNorm': False,
            'colNorm': False,
            'rmDCoffset': False,
            'inSR': None, 
            'PCEN': True,
            'fmin': 150}

h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_SNR_PCEN.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest, parms=AudioParms)


# Create the database for the the validation data
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat2.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]

h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\MalahatBalanced_melSpec_8khz_PCEN.h5'
Eco.create_hdf5_dataset(annotations=annot_val,
                        hdf5_filename= h5_file_validation, parms = AudioParms)


#%%


# Create the batch loader which will report the size
trainLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                           trainTest = 'train', batch_size=164,  n_classes=7,  
                           minFreq=0)

testLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
                           trainTest = 'test', batch_size=164,  n_classes=7,  
                           minFreq=0)

valLoader =  Eco.BatchLoader2(h5_file_validation, 
                           trainTest = 'train', batch_size=164,  n_classes=7,  
                           minFreq=0)

# get the data size
valLoader.specSize
num_classes =7
input_shape = (128, 126, 1)
#%% Check generators

for batch in trainLoader:
    print(f"Batch: {batch}")  # Inspect what `batch` contains
    if batch is None:
        print("Found None in trainLoader")
    else:
        # Check the type of `batch` and its length
        print(f"Type of batch: {type(batch)}")
        print(f"Length of batch: {len(batch)}")

        # Ensure it is unpackable into `x` and `y`
        if isinstance(batch, tuple) and len(batch) == 2:
            x, y = batch
            if x is None or y is None:
                print("Found None in batch")
        else:
            print("Batch format is incorrect")



#%%

# Create, compile, and train model
model = Eco.ResNet18(input_shape, num_classes)
model = Eco.compile_model(model)
Eco.train_model(model, trainLoader, testLoader, epochs=1, tensorBoard=True)
model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_PCEN.keras')
#%%


########################################################################

confResults = Eco.batch_conf_matrix(loaded_model = model, val_batch_loader = valLoader) 
                              

########################################################################


# Now run it with the pipeline



    # # Load Keras model
    # model_path = 'C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_SNR.keras'
    # model = load_model(model_path)

    # # Spectrogram parameters
    # audio_params = {
    #     'clipDur': 2,
    #     'outSR': 16000,
    #     'nfft': 512,
    #     'hop_length': 102,
    #     'spec_type': 'mel',  # Assuming mel spectrogram is used
    #     'rowNorm': True,
    #     'colNorm': True,
    #     'rmDCoffset': True,
    #     'inSR': None
    # }
    
    # # Example detection thresholds (adjust as needed)
    # detection_thresholds = {
    #     0: 0.8,  # Example threshold for class 0
    #     1: 0.8,  # Example threshold for class 1
    #     2: 0.9,  # Example threshold for class 2
    #     3: 0.8,  # Example threshold for class 3
    #     4: 0.8,  # Example threshold for class 4
    #     5: 0.8,  # Example threshold for class 5
    #     6: 0.9   # Example threshold for class 6
    # }
    
    # class_names = {
    #     0: 'Abiotic',
    #     1: 'BKW',
    #     2: 'HW',
    #     3: 'NRKW',
    #     4: 'Offshore',
    #     5: 'SRKW',
    #     6: 'Und Bio'
    # }
    
    # # Example usage:
    # folder_path = 'C:\\TempData\\Malahat\\STN3\\20151028'
    
    # # Initialize the AudioProcessor with your model and detection thresholds
    # processor = Eco.AudioProcessor(folder_path=folder_path, model=model,
    #                            detection_thresholds=detection_thresholds, 
    #                            class_names=class_names,    
    #                            params = audio_params,
    #                            overlap=0.25)
    
    # # Process all audio files in the directory
    # processor.process_all_files()
























# # Load the hdf5 file and create a histogram of SNR values from the hdf5
# hf = h5py.File(h5_file, 'r')

# # Extract all 'SNR' values from 'train' group using list comprehension
# train_snr_values = np.array([hf['train'][key]['SNR'][()] for key in hf['train']])

# # Close the HDF5 file
# hf.close()

# # Create histogram using Matplotlib
# plt.figure(figsize=(10, 6))
# plt.hist(train_snr_values.flatten(), bins=30, alpha=0.5, label='Train SNR')

# plt.xlabel('SNR Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of SNR Values in Train Group')
# plt.legend()

# plt.grid(True)
# plt.show()



