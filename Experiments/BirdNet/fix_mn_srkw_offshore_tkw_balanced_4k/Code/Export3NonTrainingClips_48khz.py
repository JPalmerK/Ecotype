# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:18:42 2024

@author: kaity
"""
import numpy as np
import pandas as pd
import librosa
import os
import soundfile as sf
# Create data for Birdnet
annotations = pd.read_csv("C:\\Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\fix_mn_srkw_offshore_tkw_balanced_4k\\Data\\nonTrainingData.csv")


# Check the lables look good
annotations.Label.value_counts()


# Spectrogram parameters
params = {
    'clipDur': 3,
    'outSR': 48000,
    'fmin': 0
}



# Iterate over each annotation
for idx, row in annotations.iloc[0:].iterrows():
    file_path = row['FilePath']
    start_time = row['FileBeginSec']
    end_time = row['FileEndSec']
    dep = row['Dep']
    provider = row['Provider']
    utc = row['UTC']
    speciesFolder = row['Label']
    SoundFilee = row['Soundfile']
    
    # Define the base output directory and the species-specific folder
    base_output_dir = "C:\\TempData\\threeSecClips_non_training_TKWCalls_fixed\\"
    output_folder = os.path.join(base_output_dir, speciesFolder)
    
    # Create the species folder if it doesn't exist
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder, exist_ok=True)
    
    # Load and process the audio segment
    file_duration = librosa.get_duration(path=file_path)
    duration = end_time - start_time
    center_time = start_time + (duration / 2.0)
    new_start_time = center_time -( params['clipDur'] / 2)
    new_end_time = center_time + (params['clipDur'] / 2)
    
    if new_end_time - new_start_time < params['clipDur']:
        pad_length = params['clipDur'] - (new_end_time - new_start_time)
        new_start_time = max(0, new_start_time - pad_length / 2.0)
        new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
    
    new_start_time = max(0, min(new_start_time, file_duration - params['clipDur']))
    new_end_time = max(params['clipDur'], min(new_end_time, file_duration))
    
    
    
    audio_data, sample_rate = librosa.load(file_path, 
                                           sr=params['outSR'], 
                                           offset=new_start_time,
                                           duration=params['clipDur'],
                                           mono=False)
    # Determine the number of channels
    num_channels = audio_data.shape[0] if audio_data.ndim > 1 else 1
    
    # Retain only the first channel if there are multiple
    if num_channels > 1:
        print(f"Audio has {num_channels} channels. Retaining only the first channel.")
        audio_data = audio_data[0]
    
    # Scale between 0 and 1 
    audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    # Define the output file path
    clipName = f"{SoundFilee}_{provider}_{dep}_{str(idx)}.wav"
    output_file_path = os.path.join(output_folder, clipName)
    
    # Export the audio clip
    sf.write(output_file_path, audio_data, sample_rate)
    print(f"Saved {output_file_path}")



processor = Eco.BirdNetPredictor(model_path, label_path, MN_DCLDE_clips)
HW_testData_DCLDE_birdnet6, DCLDE_HW_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
DCLDE_HW_birdnet06['Class'] = DCLDE_HW_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
DCLDE_HW_birdnet06['Truth']= 'HW'

processor = Eco.BirdNetPredictor(model_path, label_path, SRKW_DCLDE_clips)
RKW_testData_DCLDE_birdnet6, DCLDE_srkw_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
DCLDE_srkw_birdnet06['Class'] = DCLDE_srkw_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
DCLDE_srkw_birdnet06['Truth']= 'SRKW'

processor = Eco.BirdNetPredictor(model_path, label_path, TKW_DCLDE_clips)
TKW_DCLDE_Data_birdnet6, DCLDE_tkw_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
DCLDE_tkw_birdnet06['Class'] = DCLDE_tkw_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
DCLDE_tkw_birdnet06['Truth']= 'TKW'


processor = Eco.BirdNetPredictor(model_path, label_path, OKW_DCLDE_clips)
OKW_testData_DCLDE_birdnet6, DCLDE_OKW_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
DCLDE_OKW_birdnet06['Class'] = DCLDE_OKW_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
DCLDE_OKW_birdnet06['Truth']= 'OKW'


processor = Eco.BirdNetPredictor(model_path, label_path, Background_DCLDE_clips)
BG_testData_DCLDE_birdnet6, DCLDE_BG_birdnet06 = processor.batch_process_audio_folder(output_csv, return_raw_scores=True)
DCLDE_BG_birdnet06['Class'] = DCLDE_BG_birdnet06[['SRKW','OKW', 'HW', 'TKW' ]].idxmax(axis=1)
DCLDE_BG_birdnet06['Truth']= 'Backgorund'




EvalDat_DCLDE  = pd.concat([
    DCLDE_srkw_birdnet06, 
    DCLDE_tkw_birdnet06, 
    DCLDE_HW_birdnet06,
    DCLDE_OKW_birdnet06,
    DCLDE_BG_birdnet06 ])

EvalDat_DCLDE['Score'] = EvalDat_DCLDE.apply(lambda row: row[row['Class']], axis=1)

EvalDat_DCLDE['FP'] = EvalDat_DCLDE['Class'] !=  EvalDat_DCLDE['Truth']
FPData_DCLDEbirdnet06= EvalDat_DCLDE[EvalDat_DCLDE['FP'] == True]

# Get the 90% probability thresholds
aa, bb =plot_logistic_fit_with_cutoffs(DCLDE_HW_birdnet06, score_column="HW", 
                                   class_column="Class", truth_column="Truth")
HW_90_cutoff = find_cutoff(aa, 1)[0]# 95th percentile 



aa, bb = plot_logistic_fit_with_cutoffs(DCLDE_tkw_birdnet06, score_column="TKW", 
                                   class_column="Class", truth_column="Truth")
TKW_90_cutoff = find_cutoff(aa, 1)[0] # 95th percentile 


aa, bb = plot_logistic_fit_with_cutoffs(DCLDE_srkw_birdnet06, score_column="SRKW", 
                                   class_column="Class", truth_column="Truth")
SRKW_90_cutoff = find_cutoff(aa, 1)[0]# 95th percentile 

aa, bb = plot_logistic_fit_with_cutoffs(DCLDE_OKW_birdnet06, score_column="OKW", 
                                   class_column="Class", truth_column="Truth")
OKW_90_cutoff = find_cutoff(aa, 1)[0]# 95th percentile 



# Using a dictionary threshold:
custom_thresholds = {
    "SRKW": SRKW_90_cutoff,
    "TKW": TKW_90_cutoff,
    'HW':HW_90_cutoff,
    'OKW': OKW_90_cutoff}


plot_one_vs_others_pr(EvalDat_DCLDE,
                      relevant_classes=['SRKW', 'TKW', 'OKW', 'HW'], 
                      class_colors=None,
                      titleStr="One-vs-Others Precision–Recall DCLDE Curve")


class_colors = {
    'SRKW': '#1f77b4',   # Blue
    'TKW': '#ff7f0e',   # Orange
    'HW': '#2ca02c',  # Green
    'OKW':'#e377c2'}   


relevant_classes = ['SRKW', 'TKW', 'OKW', 'HW']
roc_results_06 = plot_one_vs_others_roc(FPData_DCLDEbirdnet06,  
                                     EvalDat_DCLDE,
                                     #relevant_classes = relevant_classes,
                                     titleStr= "One-vs-Others ROC Birdnet 6", 
                                     class_colors= class_colors)


cm_df = plot_confusion_matrix(EvalDat_DCLDE, threshold=custom_thresholds)