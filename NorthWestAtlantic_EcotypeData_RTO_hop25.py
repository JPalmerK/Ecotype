# # -*- coding: utf-8 -*-
# """
# Created on Fri Jun 21 10:14:13 2024

# @author: kaity

# This is a trial run of code using the separated audio repreentations. The 
# intention is to produce a full pipeline of audio processing


# Training data included all datasets
# All frequencies
# KW labels for resident tranisent and offshore only all other labels


# """





import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model
import librosa
import os

# np.random.seed(seed=2)

# # New more balanced train/test dataset so load that then create the HDF5 database
# # annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain2_Edrive.csv")
# # annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest2_Edrive.csv")

# # New more balanced train/test dataset so load that then create the HDF5 database
# annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/RTO_train.csv")
# annot_test = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/RTO_test.csv")
# annot_val = pd.read_csv('C:/Users/kaity/Documents/GitHub/Ecotype/RTO_malahat.csv')


# AllAnno = pd.concat([annot_train, annot_test], axis=0)
# AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# # Shuffle the DataFrame rows for debugging
# AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)


# # Function to create and save a spectrogram plot
# def makePlot(spectrogram, save_path):
#     if isinstance(spectrogram, np.ndarray):
#         spectrogram = spectrogram.astype(np.float32)
#     else:
#         raise TypeError("Spectrogram should be a NumPy array.")
#     plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
#     plt.imshow(np.array(spectrogram), 
#                 aspect='auto', 
#                 origin='lower', 
#                 cmap='viridis')  # Adjust parameters as needed
#     plt.colorbar(label='Intensity')
#     plt.xlabel('Time')
#     plt.ylabel('Frequency Bin')
#     plt.title('Spectrogram')
#     plt.tight_layout()
#     plt.savefig(save_path, format='png')  # Save the figure
#     plt.close()  # Close the plot to free up memory


    
# #%% Test the parameters by making example spectrograms


# # Set up audio parameters
# AudioParms = {
#             'clipDur': 3,
#             'outSR': 8000,
#             'nfft': 512,
#             'hop_length':256,
#             'spec_type': 'mel',  
#             'spec_power':2,
#             'rowNorm': False,
#             'colNorm': False,
#             'rmDCoffset': False,
#             'inSR': None, 
#             'PCEN': True,
#             'fmin': 0}




# #%% Make the HDF5 files for training testing and evaluation


# h5_fileTrainTest = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Balanced_melSpec_8khz_PCEN_RTW.h5'
# Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_fileTrainTest_hop25, 
#                         parms=AudioParms)


# # Create the database for the the validation data
# annot_val['traintest'] = 'Train'
# h5_file_validation = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Malahat_Balanced_melSpec_8khz_PCEN_RTW.h5'
# Eco.create_hdf5_dataset(annotations=annot_val,
#                         hdf5_filename= h5_file_validation_hop25, 
#                         parms = AudioParms)


# #%%


# # Create the batch loader which will report the size
# trainLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
#                            trainTest = 'train', batch_size=164,  n_classes=6,  
#                            minFreq=0)

# testLoader =  Eco.BatchLoader2(h5_fileTrainTest, 
#                            trainTest = 'test', batch_size=164,  n_classes=6,  
#                            minFreq=0)

# valLoader =  Eco.BatchLoader2(h5_file_validation, 
#                            trainTest = 'train', batch_size=164,  n_classes=6,  
#                            minFreq=0,   return_data_labels = False)

# # get the data size
# trainLoader.specSize
# num_classes =6
# input_shape = (128, 94, 1)

# #%% Train the detector



# # Create, compile, and train model
# model = Eco.ResNet18(input_shape, num_classes)
# model = Eco.compile_model(model)
# Eco.train_model(model, trainLoader, testLoader, epochs=10)
# model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_8khz_RTOt_hop25.keras')
# #%%


# label_dict = dict(zip(AllAnno['label'], AllAnno['Labels']))

# h5_file_validation1 = 'C:/Users/kaity/Documents/GitHub/Ecotype/HDF5 Files\\Malahat_Balanced_melSpec_8khz_PCEN_RTW.h5'
# model1 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_8khz_RTO.keras')

# valLoader1 =  Eco.BatchLoader2(h5_file_validation, 
#                            trainTest = 'train', batch_size=164,  n_classes=6,  
#                            minFreq=0,   return_data_labels = False)


# # Instantiate the evaluator
# evaluator = Eco.ModelEvaluator( loaded_model=model1, val_batch_loader = valLoader1, label_dict =label_dict)

# # Run the evaluation (only once)
# evaluator.evaluate_model()

# # Get the various outputs for model checking
# conf_matrix_df, conf_matrix_raw, accuracy = evaluator.confusion_matrix()
# scoreDF = evaluator.score_distributions()
# pr_curves = evaluator.precision_recall_curves()

############################################################################
# Run model on Eval data
###########################################################################
from keras.models import load_model
import pandas as pd
import EcotypeDefs as Eco

# Load the keras model
model1 = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_Resnet18_8khz_RTO.keras')
annot_test = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/RTO_test.csv")

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\20210728\\\\'

label_dict = dict(zip(annot_test['label'], annot_test['Labels']))


# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 0.25,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 0.25,  # Example threshold for class 5
}

class_names = {
    0: 'AB',
    1: 'HW',
    2: 'RKW',
    3: 'OKW',
    4: 'TKW',
    5: 'UndBio'
}


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

# Initialize the processor with custom parameters


processor = Eco.AudioProcessor4(folder_path=folder_path, 
                                segment_duration=3, overlap=0, 
                                params=AudioParms, model=model1, 
                                class_names=class_names,
                                detection_thresholds=detection_thresholds,
                                selection_table_name="detections_20241126.txt")


# Process all files in the directory
processor.process_all_files()

# # Initialize the AudioProcessor with your model and detection thresholds
# processor = Eco.AudioProcessorSimple(folder_path=folder_path,  
#                                      model=model1,
#                            detection_thresholds=detection_thresholds, 
#                            class_names=class_names,
#                            params= AudioParms,
#                            table_type="sound",
#                            overlap=0.25)

# # Process all audio files in the directory
# processor.process_all_files()










