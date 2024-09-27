import os
import soundfile as sf # For reading audio files
import numpy as np
from EcotypeDefs import create_spectrogram  # Assuming this function is defined in EcotypeDefs.py
import sounddevice as sd  # For capturing real-time audio


class AudioProcessor:
    def __init__(self, folder_path=None, segment_duration=2.0, overlap=1.0, 
                 params=None, model=None, detection_thresholds=None,
                 selection_table_name="detections.txt", class_names =None):
        self.folder_path = folder_path
        self.audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))] if folder_path else []
        self.segment_duration = segment_duration  # Duration of each segment in seconds
        self.overlap = overlap  # Overlap between segments in seconds
        self.params = params if params is not None else {
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
        self.model = model
        self.model_input_shape = model.input_shape[1:] if model else None
        
        # Initialize variables for real-time streaming
        self.buffer = np.array([], dtype='float32')  # Buffer to accumulate audio chunks
        self.sr = None  # Sample rate of the incoming audio stream
        
        # Detection thresholds for each class
        self.detection_thresholds = detection_thresholds if detection_thresholds else {  # Example thresholds
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 0.5,
            6: 0.5,
        }
        
        # Dictionary to map class IDs to names
        self.class_names = class_names if class_names else {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio',
        }
        
        # Dictionary to track ongoing detections
        self.ongoing_detections = {class_id: None for class_id in range(len(self.detection_thresholds))}
        
        # Selection table file name
        self.selection_table_name = selection_table_name
        
        # Initialize or create the selection table file
        self.init_selection_table()

        # Counter for tracking the number of detections
        self.detection_counter = 0

    def init_selection_table(self):
        # Create or overwrite the selection table file with headers
        with open(self.selection_table_name, 'w') as f:
            f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tClass\n")
    
    def load_audio_generator(self, filename):
        audio_path = os.path.join(self.folder_path, filename)
        y, sr = sf.read(audio_path, dtype='float32')
        self.params['inSR'] = sr  # Update the input sample rate
        return y, sr

    def create_segments(self, y, sr):
        # Create audio segments
        
        segment_length = int(self.segment_duration * sr)
        overlap_length = int(self.overlap * sr)
        for start in range(0, len(y) - segment_length + 1, segment_length - overlap_length):
            yield y[start:start + segment_length], start / sr  # Yield segment and start time in seconds

    def create_spectrogram(self, y, sr):
        # Create the audio representation using the same parameters as were
        # used to build the HDF5 file in training/testing/ validation
        
        if 'inSR' not in self.params or self.params['inSR'] is None:
            raise ValueError("Input sample rate 'inSR' must be set before creating spectrograms.")
        expected_time_steps = self.model_input_shape[1] if self.model_input_shape else None
        
        spectrogram = create_spectrogram(y, return_snr=False, **self.params)
        
        if expected_time_steps and spectrogram.shape[1] < expected_time_steps:
            spectrogram = np.pad(spectrogram, ((0, 0), 
                                               (0, expected_time_steps - 
                                                spectrogram.shape[1])), 
                                 mode='constant')
        
        elif spectrogram.shape[1] > expected_time_steps:
            spectrogram = spectrogram[:, :expected_time_steps]
        
        return spectrogram

    def process_audio_chunk(self, chunk, segment_start_time):
        self.buffer = np.append(self.buffer, chunk)
        if self.sr is None:
            #  get sample rate
            self.sr = sd.query_devices(None, 'input')['default_samplerate']  
        
        segment_length = int(self.segment_duration * self.sr)
        overlap_length = int(self.overlap * self.sr)
        
        while len(self.buffer) >= segment_length:
            segment = self.buffer[:segment_length]
            self.buffer = self.buffer[segment_length - overlap_length:]
            
            spectrogram = self.create_spectrogram(segment, self.sr)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
            predictions = self.model.predict(spectrogram)[0]  # Assuming batch size 1
            
            # Process predictions for each class
            for class_id, prediction_score in enumerate(predictions):
                detection_threshold = self.detection_thresholds[class_id]
                
                if prediction_score >= detection_threshold:
                    # Start or merge ongoing detection
                    if self.ongoing_detections[class_id] is None:
                        # Start new detection
                        self.ongoing_detections[class_id] = {
                            'start_time': segment_start_time,
                            'end_time': segment_start_time + self.segment_duration,
                            'class_id': class_id,
                        }
                    else:
                        # Merge with ongoing detection
                        self.ongoing_detections[class_id]['end_time'] = 
                        segment_start_time + self.segment_duration
                else:
                    # End ongoing detection
                    if self.ongoing_detections[class_id] is not None:
                        # Output the detection to selection table
                        self.output_detection(self.ongoing_detections[class_id])
                        self.ongoing_detections[class_id] = None
    
    def output_detection(self, detection):
        # Increment detection counter
        self.detection_counter += 1
        
        # Write detection to selection table file
        with open(self.selection_table_name, 'a') as f:
            selection = self.detection_counter
            start_time = detection['start_time']
            end_time = detection['end_time']
            class_id = detection['class_id']
            class_name = self.class_names[class_id]
            
            # Write detection details to file
            f.write(f"{selection}\tSpectrogram\t1\t{start_time:.6f}\t{end_time:.6f}\t0\t8000\t{class_name}\n")

            
if __name__ == "__main__":
    from keras.models import load_model

    # Load Keras model
    model = load_model('C:/Users/kaity/Documents/GitHub/Ecotype/Models\\Balanced_melSpec_8khz_SNR.keras')

    # Spectrogram parameters
    AudioParms = {
        'clipDur': 2,
        'outSR': 16000,
        'nfft': 512,
        'hop_length': 102,
        'spec_type': 'mel',  # Assuming mel spectrogram is used
        'rowNorm': True,
        'colNorm': True,
        'rmDCoffset': True,
        'inSR': None
    }
    
    

    # Example usage for testing and debugging - these files have SRKWs in them
    folder_path = 'C:\\TempData\\Malahat\\STN3\\20151028'
    # Example usage:
    # Example detection thresholds (adjust as needed)
    detection_thresholds = {
        0: 0.8,  # Example threshold for class 0
        1: 0.5,  # Example threshold for class 1
        2: 0.5,  # Example threshold for class 2
        3: 0.8,  # Example threshold for class 3
        4: 0.8,  # Example threshold for class 4
        5: 0.5,  # Example threshold for class 5
        6: 0.8   # Example threshold for class 6
    }
    
    
    class_names =  {
            0: 'Abiotic',
            1: 'BKW',
            2: 'HW',
            3: 'NRKW',
            4: 'Offshore',
            5: 'SRKW',
            6: 'Und Bio'}
    # Initialize the AudioProcessor with your model and detection thresholds
    processor = AudioProcessor(folder_path=folder_path, model=model,
                               detection_thresholds=detection_thresholds, 
                               class_names= class_names)
    
    # Process each audio file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            print(f"Processing file: {filename}")
            
            # Load audio file and process segments
            audio_generator, sr = processor.load_audio_generator(filename)
            segments = processor.create_segments(audio_generator, sr)
    
            # Process each segment
            for segment, segment_start_time in segments:
                processor.process_audio_chunk(segment, segment_start_time)