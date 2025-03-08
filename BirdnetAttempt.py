# -*- coding: utf-8 -*-



from pathlib import Path

from birdnet import SpeciesPredictions, get_species_from_file, predict_species_within_audio_file
from birdnet.models.v2m4 import CustomAudioModelV2M4TFLite, AudioModelV2M4Protobuf, get_custom_device
from birdnet import predict_species_within_audio_files_mp

audio_path = Path("C:/TempData\\AllData_forBirdnet\\MalahatValidation\\")
species_path = Path("C:/Users\\kaity\\Documents\\GitHub\\Ecotype\\Experiments\\BirdNet\\KWs_nonKW\\CustomClassifier_Labels.txt")

# Create a tuple of all audio file paths in the directory and subdirectories
audio_files = tuple(audio_path.rglob("*.wav"))

# Print the result
print(audio_files)


# create model instance for v2.4 with language 'en_us'
model = get_custom_device()
model = AudioModelV2M4Protobuf(language="en_us")

# predict only the species from this file
custom_species = get_species_from_file(species_path)

predictions = SpeciesPredictions(predict_species_within_audio_file(
  audio_files[1],
  species_filter=custom_species,
  custom_model=model,
  chunk_overlap_s=2
))


# get most probable prediction at time interval 0s-3s
prediction, confidence = list(predictions[(0.0, 3.0)].items())[0]