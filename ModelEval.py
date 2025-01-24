# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:04:22 2025

@author: kaity
"""


from keras.models import load_model
import EcotypeDefs as Eco
import numpy as np
import pandas as pd

# Models and parameters
mod_1 = load_model('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\HumpbackBalanced_8khz\\output_BalancedMn_8khz_MnBalanced_8khz_Resnet18_RTO_batchNorm.keras')

AudioParms_mod1 = {
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


# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 2,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 2,  # Example threshold for class 5
}


class_names={0: 'AB', 1: 'HW', 2: 'RKW', 3: 'OKW', 4: 'TKW', 5: 'UndBio'}


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod1, 
    model=mod_1, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
RKW_testData = fasterProcesser.get_detections()



# Figure out the false positives
MN_testData['FP'] = MN_testData['Class'] != 'HW'
MN_testData['Class'].value_counts()


AB_testData['FP'] = AB_testData['Class'] != 'AB'

UndBio_testData['FP'] = UndBio_testData['Class'] != 'UndBio'
UndBio_testData['Class'].value_counts()

RKW_testData['FP'] = RKW_testData['Class'] != 'RKW'
RKW_testData['Class'].value_counts()



ALLData = pd.concat([MN_testData, AB_testData,UndBio_testData, RKW_testData])
FPData = ALLData[ALLData['FP'] == True]



import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData['Class'].unique()}

# For each class and threshold, calculate the number of false positives
for cls in FPData['Class'].unique():
    df_class = FPData[FPData['Class'] == cls]
    false_positives = []
    
    for threshold in thresholds:
        # Apply the threshold to classify predictions
        df_class['Predicted'] = df_class['Score'] >= threshold
        # Calculate false positives (Predicted = True but True_Label = Negative)
        false_positives_count = len(df_class[(df_class['Predicted'] == True)])
        false_positives.append(false_positives_count)
    
    false_positives_by_class[cls] = false_positives

# Plotting the number of false positives for each class as a function of the threshold
plt.figure(figsize=(10, 6))
for cls, false_positives in false_positives_by_class.items():
    sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# Add labels and title
plt.xlabel("Threshold Score")
plt.ylabel("Number of False Positives")
plt.title("False Positives as a Function of Threshold Score by Class")
plt.legend(title="Class")
plt.grid(True)
plt.show()



##############################################################################
# Resnet 50
##############################################################################


# Models and parameters
mod_2 = load_model('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\HumpbackBalanced_16khz\\output\\Resnet50\\output_BalancedMn_16khz_Resnet50_MnBalanced_16khz_Resnet50_20241228_042445.keras')
                   
                   
AudioParms_mod2 = {
    'clipDur': 3,
    'outSR': 16000,
    'nfft': 1024,
    'hop_length': 102,
    'spec_type': 'mel',
    'spec_power': 2,
    'rowNorm': False,
    'colNorm': False,
    'rmDCoffset': False,
    'inSR': None,
    'PCEN': True,
    'fmin': 0,
    'min_freq': None,
    'returnDB': True,
    'PCEN_power': 31,
    'time_constant': 0.8,
    'eps': 1e-6,
    'gain': 0.08,
    'power': 0.25,
    'bias': 10,
    'fmax': 16000,
    'Scale Spectrogram': False}

# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 2,  # Example threshold for class 0
    1: 0.25,  # Example threshold for class 1
    2: 0.25,  # Example threshold for class 2
    3: 0.25,  # Example threshold for class 3
    4: 0.25,  # Example threshold for class 4
    5: 2,  # Example threshold for class 5
}


class_names={0: 'AB', 1: 'HW', 2: 'RKW', 3: 'OKW', 4: 'TKW', 5: 'UndBio'}


# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Humpback\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
MN_testData_mod2 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\Abiotic\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
AB_testData_mod2 = fasterProcesser.get_detections()

# Audio Files to run
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SanctSound\\UnkBio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
UndBio_testData_mod2 = fasterProcesser.get_detections()

# SMRU Resident killer whales
folder_path = 'C:\\TempData\\DCLDE_EVAL\\SMRU\\Audio\\'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
RKW_testData_mod2 = fasterProcesser.get_detections()


folder_path ='C:\\TempData\\AllData_forBirdnet\\MalahatValidation\\TKW'
fasterProcesser = Eco.AudioProcessor5(
    folder_path=folder_path, 
    segment_duration=AudioParms_mod1['clipDur'], 
    overlap=0, 
    params=AudioParms_mod2, 
    model=mod_2, 
    class_names=class_names,
    detection_thresholds=detection_thresholds,
    selection_table_name="Test.txt")
fasterProcesser.process_all_files()
TKW_testData_mod2 = fasterProcesser.get_detections()



# Figure out the false positives
MN_testData_mod2['FP'] = MN_testData_mod2['Class'] != 'HW'
MN_testData_mod2['Class'].value_counts()

AB_testData_mod2['FP'] = AB_testData_mod2['Class'] != 'AB'

UndBio_testData_mod2['FP'] = UndBio_testData_mod2['Class'] != 'UndBio'
UndBio_testData_mod2['Class'].value_counts()

RKW_testData_mod2['FP'] = RKW_testData_mod2['Class'] != 'RKW'
RKW_testData_mod2['Class'].value_counts()

TKW_testData_mod2['FP'] = TKW_testData_mod2['Class'] != 'TKW'
TKW_testData_mod2['Class'].value_counts()



ALLData_mod2 = pd.concat([MN_testData_mod2, AB_testData_mod2,
                     UndBio_testData_mod2, RKW_testData_mod2, 
                     TKW_testData_mod2])
FPData_mod2 = ALLData_mod2[ALLData_mod2['FP'] == True]



import matplotlib.pyplot as plt
import seaborn as sns


# Set a range of threshold scores
thresholds = np.linspace(0.35, 1, 100)  # Adjust the number of thresholds as needed

# Prepare a list to hold false positive counts for each class
false_positives_by_class = {cls: [] for cls in FPData_mod2['Class'].unique()}

# For each class and threshold, calculate the number of false positives
for cls in FPData_mod2['Class'].unique():
    df_class = FPData_mod2[FPData_mod2['Class'] == cls]
    false_positives = []
    
    for threshold in thresholds:
        # Apply the threshold to classify predictions
        df_class['Predicted'] = df_class['Score'] >= threshold
        # Calculate false positives (Predicted = True but True_Label = Negative)
        false_positives_count = len(df_class[(df_class['Predicted'] == True)])
        false_positives.append(false_positives_count)
    
    false_positives_by_class[cls] = false_positives

# Plotting the number of false positives for each class as a function of the threshold
plt.figure(figsize=(10, 6))
for cls, false_positives in false_positives_by_class.items():
    sns.lineplot(x=thresholds, y=false_positives, label=cls, marker='o')

# Add labels and title
plt.xlabel("Threshold Score")
plt.ylabel("Number of False Positives")
plt.title("False Positives as a Function of Threshold Score by Class")
plt.legend(title="Class")
plt.grid(True)
plt.show()

