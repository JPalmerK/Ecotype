import os
import soundfile as sf
#import librosa
import numpy as np
from EcotypeDefs import create_spectrogram  # Assuming this function is defined in EcotypeDefs.py

class AudioProcessor:
    def __init__(self, folder_path, segment_duration=2.0, overlap=1.0, params=None):
        self.folder_path = folder_path
        self.audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        self.segment_duration = segment_duration  # Duration of each segment in seconds
        self.overlap = overlap  # Overlap between segments in seconds
        self.model_input_shape = None  # To store model's input shape
        
        # Initialize numpy array for predictions and list for time since start
        self.all_predictions = None
        self.time_since_start = []

        # Set default parameters for create_spectrogram
        if params is None:
            params = {
                'outSR': 16000,
                'clipDur': segment_duration,
                'nfft': 512,
                'hop_length': 3200,
                'spec_type': 'mel',  # Assuming mel spectrogram is used
                'rowNorm': True,
                'colNorm': True,
                'rmDCoffset': True,
                'inSR': None
            }
        self.params = params

    def load_audio_generator(self, filename):
        audio_path = os.path.join(self.folder_path, filename)
        y, sr = sf.read(audio_path, dtype='float32')
        
        # Update the input sample rate
        self.params['inSR'] = sr
        return y, sr

    def create_segments(self, y, sr):
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        segments = []
        for start in range(0, len(y) - segment_length + 1, segment_length - overlap_length):
            segment = y[start:start + segment_length]
            segments.append(segment)
        return segments

    def create_spectrogram(self, y, sr):
        if self.model_input_shape is None:
            raise ValueError("Model input shape must be set before creating spectrograms.")
        
        # Ensure spectrogram matches the model's input shape
        expected_time_steps = self.model_input_shape[1]
    
        # Call create_spectrogram function with parameters
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        
        # Resize or pad to match expected input shape
        if spectrogram.shape[1] < expected_time_steps:
            # Pad spectrogram if it's shorter than expected time steps
            spectrogram = np.pad(spectrogram, 
                                 ((0, 0), 
                                  (0, expected_time_steps - spectrogram.shape[1])), 
                                 mode='constant')
        elif spectrogram.shape[1] > expected_time_steps:
            # Trim spectrogram if it's longer than expected time steps
            spectrogram = spectrogram[:, :expected_time_steps]
        
        return spectrogram

    def predict_segments(self, segments, model, sr):
        if self.model_input_shape is None:
            self.model_input_shape = model.input_shape[1:]  # Skip batch size dimension
    
        # Initialize predictions array and time list if not already initialized
        if self.all_predictions is None:
            self.all_predictions = np.empty((len(segments), model.output_shape[1]))
            self.time_since_start = [0] * len(segments)  # Placeholder for time since start
    
        time_accumulated = 0  # Track the accumulated time
        segment_duration_samples = int(self.segment_duration * sr)
        overlap_samples = int(self.overlap * sr)
    
        for idx, segment in enumerate(segments):
            spectrogram = self.create_spectrogram(segment, sr)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            prediction = model.predict(spectrogram)
    
            # Update predictions array
            self.all_predictions[idx] = prediction
    
            # Update time since start
            if idx > 0:
                time_accumulated +=  overlap_samples  # Add overlap for all except the first segment
            self.time_since_start[idx] = time_accumulated / sr  # Convert to seconds
    
        return self.all_predictions, self.time_since_start

    def create_detections(self, all_predictions, 
                          thresholds=[0.8, 0.8, 0.8, 0.8, 0.8],
                          min_duration=1.0):
        detections = {i: [] for i in range(len(thresholds))}   # Detections for each class
    
        segment_duration = self.segment_duration
        for file_idx, file_preds in enumerate(all_predictions):
            for segment_idx, segment_preds in enumerate(file_preds):
                for class_index, score in enumerate(segment_preds):
                    if score > thresholds[class_index]:
                        detection_start = segment_idx * (segment_duration - self.overlap)
                        detection_end = detection_start + segment_duration
                        detections[class_index].append((detection_start, detection_end))
    
        # Post-process to merge detections and ensure minimum duration
        for class_index in range(len(detections)):
            detections[class_index] = self.merge_and_filter_detections(detections[class_index], min_duration)
    
        return detections
    
    def merge_and_filter_detections(self, detections, min_duration):
        merged_detections = []
        if len(detections) > 0:
            current_start, current_end = detections[0]
            for detection in detections[1:]:
                if detection[0] - current_end <= min_duration:
                    current_end = detection[1]
                else:
                    merged_detections.append((current_start, current_end))
                    current_start, current_end = detection
            merged_detections.append((current_start, current_end))
        return merged_detections

if __name__ == "__main__":
    
    from keras.models import load_model
    
    
    # Spectrogram parameters
    AudioParms = {
            'clipDur': 2,
            'outSR': 16000,
            'nfft': 512,
            'hop_length':102,
            'spec_type': 'mel',  # Assuming mel spectrogram is used
            'rowNorm': True,
            'colNorm': True,
            'rmDCoffset': True,
            'inSR': None}
    
    
    # Example usage for testing and debugging
    folder_path = 'C:\\TempData\\Malahat\\STN1\\20160218'
    audio_processor = AudioProcessor(folder_path, segment_duration=2.0, 
                                     overlap=0.5,  params= AudioParms)

    # Load  Keras model
    model = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\ReBalanced_melSpec_7class_8khz_wider_3.keras')


    # Process audio files segment by segment and predict
    all_predictions = []
    for filename in audio_processor.audio_files:
        # Load audio data as a generator
        audio_generator, sr = audio_processor.load_audio_generator(filename)
        
        # Create segments from the generator
        segments = audio_processor.create_segments(audio_generator, sr)
        
        # Predict on segments
        predictions = audio_processor.predict_segments(segments, model, sr)
        #all_predictions.append(predictions)

    # Create detections based on predictions for each class
    thresholds = [.8, .8, .8, .8, .8, .8, .8]  # Adjust thresholds as needed for each class
    detections = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}  # Initialize detections dictionary
    
    for class_index in range(7):  # Assuming there are 5 classes
        detections[class_index] = audio_processor.create_detections(all_predictions, 
                                                                    thresholds=thresholds)
        
        aa = audio_processor.create_detections(all_predictions, thresholds=thresholds)
        # Output detections to Raven Pro selection table for each class
        output_file = f'detections_class_{class_index}.csv'
        audio_processor.output_to_raven_pro(detections[class_index], class_index, output_file)

