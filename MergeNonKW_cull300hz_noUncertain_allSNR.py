# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:14:13 2024

@author: kaity
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import EcotypeDefs as Eco
from keras.models import load_model


# New more balanced train/test dataset so load that then create the HDF5 database
annot_train = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTrain.csv")
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/EcotypeTest.csv")

AllAnno = pd.concat([annot_train, annot_val], axis=0)
AllAnno = AllAnno[AllAnno['LowFreqHz'] < 8000]

# Remove uncertain KW calls
AllAnno = AllAnno[AllAnno['KW_certain'] !=0]

# Switch labels of non KW annotations to 0
AllAnno.loc[AllAnno['label'].isin([0, 2, 6]), 'label'] = 0


# Extract the 'labels' column into a NumPy array
original_labels = AllAnno['label'].values

# Create a mapping dictionary to map old labels to new continuous labels
unique_labels = np.unique(original_labels)
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Create new_labels array with continuous labels based on label_map
new_labels = np.array([label_map[label] for label in original_labels])

# Replace the 'labels' column in the DataFrame with new_labels
AllAnno['label'] = new_labels


# Shuffle the DataFrame rows for testing
AllAnno = AllAnno.sample(frac=1, random_state=42).reset_index(drop=True)

label_mapping_traintest = AllAnno[['label', 'Labels']].drop_duplicates()

# Create the HDF5 file
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz_kw_vsNot.h5'
Eco.create_hdf5_dataset(annotations=AllAnno, hdf5_filename= h5_file)
#
# Assuming annotations is your DataFrame containing audio segment metadata
#Eco.create_hdf5_dataset_parallel(AllAnno, 'data_parallel2.h5',  8)this will work 
# but the slowdown is with the USB connection, not parallel processing so it
# just made it slower. Sad day.





##############################################################################
# Cool now train a model
##############################################################################

import EcotypeDefs as Eco


h5_file = 'Balanced_melSpec_8khz_kw_vsNot.h5'

# Create the train and test batch loaders
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=5,    
                              minFreq=300)


test_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=5,
                           minFreq=300)

# Create the Resnet 
num_classes =5
test_batch_loader.specSize
input_shape = (116, 314, 1)

# Create and compile model
model = Eco.create_model_with_resnet(input_shape, num_classes)
model = Eco.compile_model(model)




# Train model
Eco.train_model(model, train_batch_loader, test_batch_loader, epochs=20)
model.save('C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz_kw_vsNot_300hz.keras')


# Plot model activations

# Create the train and test batch loaders
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=1, n_classes=5,    
                              minFreq=300,  return_data_labels=True)

labelval = train_batch_loader.__getitem__(0)

modelLoc ='C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz_kw_vsNot_300hz.keras'
Eco.plot_arctivations(modelLoc, train_batch_loader, valIdx =10)


############################################################################
# Create the malahat combined validataions
###########################################################################

# Create the database for the the validation data
annot_val = pd.read_csv("C:/Users/kaity/Documents/GitHub/Ecotype/Malahat.csv")
annot_val = annot_val[annot_val['LowFreqHz'] < 8000]


# Remove uncertain KW calls
annot_val = annot_val[annot_val['KW_certain'] !=0]

#current label mapping
annot_val[['label', 'Labels']].drop_duplicates()

# Switch labels of non KW annotations to 0
annot_val.loc[annot_val['label'].isin([0, 2, 6]), 'label'] = 0
annot_val.loc[annot_val['label'].isin([5]), 'label'] = 4

# check that it worked
annot_val[['label', 'Labels']].drop_duplicates()


#  Labeles changed so update
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatBalanced_melSpec_8khz_kw_vsNot1.h5'
Eco.create_hdf5_dataset(annotations=annot_val,
                        hdf5_filename= h5_file)

# Create the confusion matrix
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=1000,  n_classes=5,
                           minFreq=300)

confResults = Eco.batch_conf_matrix(loaded_model= model, 
                                val_batch_loader=val_batch_loader)

# Just check that the numbers match
label_counts = annot_val['label'].value_counts()


################################################################
# Try a bigger model. Mostly shits and giggles.
###############################################################



# Create the Resnet 
num_classes =5
test_batch_loader.specSize
input_shape = (116, 314, 1)

# Create and compile model
modelResnet18 = Eco.ResNet18(input_shape, num_classes)
modelResnet18 = Eco.compile_model(model)


# Train model
Eco.train_model(modelResnet18, train_batch_loader, test_batch_loader, epochs=20)
modelResnet18.save('C:/Users/kaity/Documents/GitHub/Ecotype/RSnet18_Balanced_melSpec_8khz_kw_vsNot_300hz.keras')


# Evaluate the model
# Create the confusion matrix
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatBalanced_melSpec_8khz_kw_vsNot1.h5'
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=1000,  n_classes=5,
                           minFreq=300)

confResults = Eco.batch_conf_matrix(loaded_model= modelResnet18, 
                                val_batch_loader=val_batch_loader)



############################################################################
# Plot what the model learned
############################################################################



# Loader
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatBalanced_melSpec_8khz_kw_vsNot1.h5'

# Create the train and test batch loaders
batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=1, n_classes=5,    
                              minFreq=300,  return_data_labels=True)

modelLoc ='C:/Users/kaity/Documents/GitHub/Ecotype/Balanced_melSpec_8khz_HWcull300Hz.keras'
Eco.plot_arctivations(modelLoc, train_batch_loader, valIdx =1)


from tensorflow.keras import models
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Activation
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import EcotypeDefs as Eco
import matplotlib.pyplot as plt



model = load_model(modelLoc)


# Example: Extracting output from the first convolutional layer
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)





dataPreds = train_batch_loader.__getitem__(0)
data = np.array(dataPreds[0])


activations = activation_model.predict(data)

for i, activation in enumerate(activations):
    # Example: Visualize the activation map for the i-th layer
    plt.matshow(activation[0, :, :, 0], cmap='viridis')  # Adjust indexing based on your layer's shape
    plt.title('Activation Map of Layer {}'.format(i))
    plt.colorbar()
    plt.show()
    
##############################################################################
