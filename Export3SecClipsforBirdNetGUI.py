# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:18:42 2024

@author: kaity
"""

import pandas as pd
import librosa
import os
import soundfile as sf
# Create data for Birdnet
annotations = pd.read_csv("C:/Users/kaity/Documents/PNW_ClassifierAnnotations.csv")

# Spectrogram parameters
params = {
    'clipDur': 3,
    'outSR': 16000,
    'fmin': 150
}




# Iterate over each annotation
for idx, row in annotations.iloc[0:].iterrows():
    file_path = row['FilePath']
    start_time = row['FileBeginSec']
    end_time = row['FileEndSec']
    dep = row['Dep']
    provider = row['Provider']
    utc = row['UTC']
    speciesFolder = row['AnnoBin']
    SoundFilee = row['Soundfile']
    
    # Define the base output directory and the species-specific folder
    base_output_dir = "C:\\TempData\\PNW_TrainingDataUnfiltered"
    output_folder = os.path.join(base_output_dir, speciesFolder)
    
    # Create the species folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load and process the audio segment
    file_duration = librosa.get_duration(path=file_path)
    duration = end_time - start_time
    center_time = (start_time + end_time) / 2.0
    new_start_time = center_time - params['clipDur'] / 2
    new_end_time = center_time + params['clipDur'] / 2
    
    if new_end_time - new_start_time < params['clipDur']:
        pad_length = params['clipDur'] - (new_end_time - new_start_time)
        new_start_time = max(0, new_start_time - pad_length / 2.0)
        new_end_time = min(file_duration, new_end_time + pad_length / 2.0)
    
    new_start_time = max(0, min(new_start_time, file_duration - params['clipDur']))
    new_end_time = max(params['clipDur'], min(new_end_time, file_duration))
    
    audio_data, sample_rate = librosa.load(file_path, 
                                           #sr=params['outSR'], 
                                           offset=new_start_time,
                                           duration=params['clipDur'])
    
    # Define the output file path
    clipName = f"{SoundFilee}_{provider}_{dep}_{str(idx)}.wav"
    output_file_path = os.path.join(output_folder, clipName)
    
    # Export the audio clip
    sf.write(output_file_path, audio_data, sample_rate)
    print(f"Saved {output_file_path}")
    
    
    
    
    
    
    
