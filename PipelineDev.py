# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:48:01 2024

@author: kaity
"""
from keras.models import load_model
import EcotypeDefs as Eco

# Now run it with the pipeline



# Load Keras model
model_path = 'C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Models\\20200818\\output_resnet18_PECN_melSpec.keras'
model = load_model(model_path)



# Example usage of audio
folder_path = 'E:\\Malahat\\STN3\\'


# Spectrogram parameters
AudioParms = {
    'clipDur': 2,
    'outSR': 16000,
    'nfft': 512,
    'hop_length': 25,
    'spec_type': 'mel',  
    'spec_power': 2,
    'rowNorm': False,
    'colNorm': False,
    'rmDCoffset': False,
    'inSR': None, 
    'PCEN': True,
    'fmin': 150
}

# Example detection thresholds (adjust as needed)
detection_thresholds = {
    0: 0.5,  # Example threshold for class 0
    1: 0.5,  # Example threshold for class 1
    2: 0.5,  # Example threshold for class 2
    3: 0.5,  # Example threshold for class 3
    4: 0.5,  # Example threshold for class 4
    5: 0.5,  # Example threshold for class 5
    6: 0.5   # Example threshold for class 6
}

class_names = {
    0: 'AB',
    1: 'BKW',
    2: 'HW',
    3: 'NRKW',
    4: 'Offshore',
    5: 'SRKW',
    6: 'UndBio'
}



# Initialize the AudioProcessor with your model and detection thresholds
processor = Eco.AudioProcessor3(folder_path=folder_path, 
                                model=model,
                           detection_thresholds=detection_thresholds, 
                           class_names=class_names,
                           params= AudioParms,
                           table_type="sound",
                           overlap=0.15)

# Process all audio files in the directory
processor.process_all_files()


#Ok now we have a selection table but we need to compare it to the annotations
# from th origional authors

import pandas
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score
import os

# Load  truth and prediction data
truth_data = pandas.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Malahat2.csv')
predictions = pandas.read_csv('C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\detections.txt',
                           sep='\t')


predictions['filename'] = predictions['Sound'].apply(lambda x: os.path.basename(x))

# rename predictions to match
predictions=predictions.rename(columns={"filename": "Soundfile",
                                        "Begin Time (S)": "FileBeginSec",
                                        'End Time (S)': "FileEndSec",
                                        'Class': 'Labels'})


predictions['CenterTime'] = predictions['FileBeginSec'] +(predictions['FileEndSec']-predictions['FileBeginSec'])/2

# Example columns: ['soundfile', 'start_time', 'end_time', 'class']
# Ensure both dataframes are sorted by soundfile and time
truth_data = truth_data.sort_values(by=['Soundfile', 'FileBeginSec'])
predictions = predictions.sort_values(by=['Soundfile', 'FileBeginSec'])


# Only keep relevent bits
truth_data = truth_data[['Soundfile', 'FileBeginSec','FileEndSec', 'CenterTime', 'Labels']]
predictions = predictions[['Soundfile', 'FileBeginSec','FileEndSec', 'CenterTime', 'Labels', 'Score']]


# Step through each class and calculate precision/recall
# Pick thresholds
# Define the thresholds from 0.5 to 0.99 with a step of 0.05
thresholds = np.round(np.arange(0.5, 1.0, 0.01),2)

# Pre-allocate a DataFrame with columns as class names and rows as thresholds
precision_df = pd.DataFrame(index=thresholds, columns=class_names.values())
recall_df = pd.DataFrame(index=thresholds, columns=class_names.values())

# Initialize the DataFrames with NaN values
precision_df[:] = np.nan
recall_df[:] = np.nan


ii =0
    
for classLabel in class_names.values():
    print(classLabel)
    detSub = predictions[predictions['Labels'] == classLabel]
    truthSub = truth_data[truth_data['Labels'] == classLabel]    
        
    for thresh in thresholds:
    
        predictionsThresh = detSub[detSub['Score']>= thresh]        
        # Create a subset of the predictions

        # Find matching predictions based on Soundfile and CenterTime within 10 seconds
        matched_results = predictionsThresh.merge(
            truthSub, 
            on='Soundfile', 
            suffixes=('_pred', '_truth')
        ).query('abs(CenterTime_pred - CenterTime_truth) <= 1.5')
        
        # Handle unmatched predictions and truth data
        unmatched_truth = truthSub[~truthSub.index.isin(matched_results.index)]
        unmatched_predictions = predictionsThresh[~predictionsThresh.index.isin(matched_results.index)]
    
        # Prepare data for precision and recall calculation
        y_true = pd.concat([matched_results['Labels_truth'], unmatched_truth['Labels']])
        y_pred = pd.concat([matched_results['Labels_pred'], pd.Series(['None'] * len(unmatched_truth))])
    
        # Add unmatched predictions as false positives
        y_true = pd.concat([y_true, pd.Series(['None'] * len(unmatched_predictions))])
        y_pred = pd.concat([y_pred, unmatched_predictions['Labels']])
    
        
        # Calculate precision and recall
        precision = precision_score(y_true, y_pred, average='binary', pos_label=classLabel, zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=classLabel, zero_division=0)
        
        # Store the precision and recall in the DataFrames
        precision_df.at[thresh, classLabel] = precision
        recall_df.at[thresh, classLabel] = recall
        
        ii = ii+1
        
        #print(f'Precision: {precision}, Recall: {recall}')
        print(f'threshold: {thresh}, class: {classLabel}')
       

    
    




# Merge based on soundfile first
# Merge based on soundfile first
merged_data = pd.merge(truth_data, predictions, on='Soundfile', suffixes=('_truth', '_pred'))




# Merge based on soundfile first
merged_data = pd.merge(truth_data, predictions, on='Soundfile', suffixes=('_truth', '_pred'))

# Calculate the absolute time difference (start_time)
merged_data['time_diff'] = np.abs(merged_data['start_time_truth'] - merged_data['FileBeginSec'])

# Define a time tolerance (e.g., 0.5 seconds)
time_tolerance = 0.5

# Filter rows where the time difference is within tolerance
matched_data = merged_data[merged_data['time_diff'] <= time_tolerance]

# Now you can calculate precision and recall based on the matched data
# Example: true positives, false positives, and false negatives

true_positive = matched_data[matched_data['class_truth'] == matched_data['class_pred']].shape[0]
false_positive = matched_data[matched_data['class_truth'] != matched_data['class_pred']].shape[0]
false_negative = truth_data.shape[0] - true_positive

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

print(f'Precision: {precision}, Recall: {recall}')