import os
import librosa
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np


# None of this worked. Some of it took many hours to run and didn't work

# Parameters
audio_folder = 'C:\\TempData\\AllData_forBirdnet\\KWsOnly\\TKW\\'
n_clusters = 20  # Modify based on how many clusters you think might be useful
n_components = 50  # Number of PCA components
batch_size = 64  # Define an appropriate batch size

# Collecting file paths
audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

# Feature extraction using a generator to save memory
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # Loading with a fixed sample rate
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=51)
    S_pcen = librosa.pcen(S * (2**31), sr=sr, hop_length=51)
    return S_pcen.astype(np.float32).flatten()  # Convert each feature to float32 immediately

# Use a generator to iterate over features
def generate_features(file_paths):
    for file in file_paths:
        yield extract_features(file)

# Instantiate IncrementalPCA
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# Handling NaN values programmatically before PCA
def impute_features(features):
    imputer = SimpleImputer(strategy='mean')  # Using mean to impute
    return imputer.fit_transform(features)

# Fit IncrementalPCA in batches using the generator
feature_generator = generate_features(audio_files)
batch_counter = 0
for batch in np.array_split(audio_files, len(audio_files) // batch_size + 1):  # Split file paths into batches
    batch_features = np.array(list(generate_features(batch)))  # Extract features for each batch
    batch_features_imputed = impute_features(batch_features)  # Impute any NaN values
    if batch_features_imputed.size > 0:
        ipca.partial_fit(batch_features_imputed)  # Incrementally fit the PCA
    batch_counter += 1
    print(f"Processed batch {batch_counter}/{len(audio_files) // batch_size + 1}")

# Transform all features in batches
features_reduced = np.vstack([ipca.transform(impute_features(np.array(list(generate_features(batch)))))
                              for batch in np.array_split(audio_files, len(audio_files) // batch_size + 1)])

# Scaling the reduced features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_reduced)

# Clustering
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(features_scaled)

# Creating DataFrame
df = pd.DataFrame({
    'filename': [os.path.basename(file) for file in audio_files],
    'filepath': audio_files,
    'cluster_id': clusters
})

# Save to CSV
df.to_csv('clustered_audio_data.csv', index=False)

print("Clustering complete. CSV file saved.")



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Further reduce dimensions for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)  # n_components can be 2 for 2D visualization
features_tsne = tsne.fit_transform(features_scaled)

# Plotting the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster ID')
plt.title('Cluster visualization with t-SNE')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()


# Base directory where all files are initially stored and where clusters are
base_dir = 'C:\\TempData\\AllData_forBirdnet\\KWsOnly\\TKW\\'





import os
import shutil
import pandas as pd

# Base directory where all files are initially stored and where clusters are
base_dir = 'C:\\TempData\\AllData_forBirdnet\\KWsOnly\\TKW\\'

# Function to check and move files
def check_and_move(row):
    # Define the source and target paths
    source_path = row['filepath']
    target_dir = os.path.join(base_dir, f'cluster_{row["cluster_id"]}')
    target_path = os.path.join(target_dir, row['filename'])

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move the file if it's not already in the correct directory
    if not os.path.exists(target_path):
        try:
            shutil.move(source_path, target_path)
            print(f"Moved file to {target_path}")
        except Exception as e:
            print(f"Failed to move {source_path}. Error: {e}")
    else:
        print(f"File already in the correct directory: {target_path}")

# Applying the function to each row in the DataFrame
df.apply(check_and_move, axis=1)



import os
import librosa
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# DBSCAN Clustering
# Standardize features before DBSCAN
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_reduced)

# Determine optimal 'eps' using Nearest Neighbors plot
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(features_scaled)
distances, indices = neighbors_fit.kneighbors(features_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]  # Considering the distance to the 2nd nearest neighbor
plt.plot(distances)
plt.title('2nd Nearest Neighbor Distance')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Configure DBSCAN
eps_value = float(input("Enter the eps value based on the elbow in the plot: "))  # User inputs eps value based on plot
min_samples_value = int(input("Enter the minimum samples value (try 5 or 10): "))  # Adjust based on your dataset
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
clusters = dbscan.fit_predict(features_scaled)

# Output the clustering results
print(f'Estimated number of clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}')
print(f'Estimated number of noise points: {list(clusters).count(-1)}')

# Optional: Create DataFrame for further analysis or to check results
df = pd.DataFrame({
    'filename': [os.path.basename(file) for file in audio_files],
    'filepath': audio_files,
    'cluster_id': clusters
})

# Display the DataFrame or save it for further analysis
print(df.head())


#%%###################################################################
# Dynamic time warping
#####################################################################

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans  # Alternatively, you could try SpectralClustering if feasible
from sklearn.preprocessing import StandardScaler
import time
import random

# PARAMETERS
audio_folder = r'C:\TempData\AllData_forBirdnet\KWsOnly\TKW\\'
n_prototypes = 100     # Number of prototype files to use for embedding
n_clusters = 10        # Number of clusters for final clustering
sr_target = 16000      # Target sample rate for audio
n_fft = 512            # FFT window size
hop_length = 51        # Hop length for spectrogram
print_interval = 100   # Print progress every 100 files

# COLLECT FILE PATHS
audio_files = [os.path.join(audio_folder, f) 
               for f in os.listdir(audio_folder) 
               if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
print(f"Found {len(audio_files)} audio files.")

# FEATURE EXTRACTION (for DTW we keep the 2D PCEN spectrogram)
def extract_features_sequence(file_path):
    y, sr = librosa.load(file_path, sr=sr_target)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Using librosa's PCEN; adjust multiplication factor as in your working code
    S_pcen = librosa.pcen(S * (2**31), sr=sr, hop_length=hop_length)
    return S_pcen.astype(np.float32)

print("Extracting features from audio files (this may take a while)...")
start_time = time.time()
features_seq = [extract_features_sequence(f) for f in audio_files]
print(f"Extracted features for {len(features_seq)} files in {time.time() - start_time:.2f} seconds.")

# SELECT PROTOTYPES
random.seed(42)
prototype_indices = random.sample(range(len(audio_files)), n_prototypes)
prototypes = [features_seq[i] for i in prototype_indices]
print(f"Selected {n_prototypes} prototypes for embedding.")

# COMPUTE DTW EMBEDDING USING FASTDTW
# For each file, compute FastDTW distance to each prototype and store as a vector.
print("Computing FastDTW distances to form the embedding...")
embedding = np.zeros((len(features_seq), n_prototypes), dtype=np.float32)
start_time = time.time()
for i, feat in enumerate(features_seq):
    if (i + 1) % print_interval == 0:
        print(f"Processing file {i+1}/{len(features_seq)}")
    for j, proto in enumerate(prototypes):
        distance, _ = fastdtw(feat, proto, dist=euclidean)
        embedding[i, j] = distance
print(f"Computed embedding in {time.time() - start_time:.2f} seconds.")

# OPTIONAL: Transform the distance embedding using an RBF (Gaussian) kernel.
# Compute sigma as the median of all distances in the embedding.
sigma = np.median(embedding)
print(f"Computed sigma = {sigma:.2f} for RBF transformation.")
embedding_rbf = np.exp(- (embedding ** 2) / (2 * sigma ** 2))

# Standardize the embedding features
scaler = StandardScaler()
embedding_scaled = scaler.fit_transform(embedding_rbf)

# CLUSTERING: Using K-means on the DTW kernel embedding.
print("Clustering using K-means on the DTW kernel embedding...")
start_time = time.time()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embedding_scaled)
print(f"Clustering complete in {time.time() - start_time:.2f} seconds.")

# Create a DataFrame with the clustering results
df = pd.DataFrame({
    'filename': [os.path.basename(f) for f in audio_files],
    'filepath': audio_files,
    'cluster_id': clusters
})
print("Sample clustering results:")
print(df.head())

# OPTIONAL: Save the results to a CSV file for later use.
df.to_csv('dtw_clustered_audio_data.csv', index=False)
print("Clustering results saved to 'dtw_clustered_audio_data.csv'.")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

# Apply t-SNE to the RBF-transformed, scaled embedding
start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  
embedding_tsne = tsne.fit_transform(embedding_scaled)
print(f"t-SNE completed in {time.time() - start_time:.2f} seconds.")

# Plot the t-SNE result, colored by cluster
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1],
                      c=clusters, cmap='viridis', s=20, alpha=0.7)
plt.colorbar(scatter, label='Cluster ID')
plt.title("Clusters Visualized in 2D (via t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()


import matplotlib.pyplot as plt

# Reduce the RBF-transformed DTW embedding to 2D using PCA
from sklearn.decomposition import PCA
pca_2d = PCA(n_components=2)
embedding_2d = pca_2d.fit_transform(embedding_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='viridis', s=20, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Clusters Visualized in 2D (via PCA)")
plt.colorbar(scatter, label="Cluster ID")
plt.show()



import os
import shutil
import pandas as pd

# Base directory where all files are initially stored and where clusters are
base_dir = 'C:\\TempData\\AllData_forBirdnet\\KWsOnly\\TKW\\'

# Function to check and move files
def check_and_move(row):
    # Define the source and target paths
    source_path = row['filepath']
    target_dir = os.path.join(base_dir, f'cluster_{row["cluster_id"]}')
    target_path = os.path.join(target_dir, row['filename'])

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move the file if it's not already in the correct directory
    if not os.path.exists(target_path):
        try:
            shutil.move(source_path, target_path)
            print(f"Moved file to {target_path}")
        except Exception as e:
            print(f"Failed to move {source_path}. Error: {e}")
    else:
        print(f"File already in the correct directory: {target_path}")



# Applying the function to each row in the DataFrame
df.apply(check_and_move, axis=1)



