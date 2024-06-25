# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:23:32 2024

@author: kaity
"""


##############################################################################
# Use the SNR and 300 hz Culled subset of the hdf5 with a custom loss function
# that prioritizes different weights
##############################################################################

import EcotypeDefs as Eco
import numpy as np
import keras


h5_file = 'Balanced_melSpec_8khz_HWcull.h5'

# Create the train and test batch loaders, this is where the frequency restriction
# happens
train_batch_loader = Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=250, n_classes=7,    
                              minFreq=300)


test_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'test', batch_size=250,  n_classes=7,
                           minFreq=300)

import EcotypeDefs as Eco
# Create the Resnet 
num_classes =7
test_batch_loader.specSize
input_shape = (116, 314, 1)

# Define class weights


# Create and compile model with custom weights
model = Eco.create_model_with_resnet(input_shape, num_classes)
model = Eco.compile_model(model, keras.losses.CategoricalFocalCrossentropy)




# Train model
Eco.train_model(model, train_batch_loader, 
                test_batch_loader,  epochs=20)


model.save('Balanced_melSpec_8khz_HWcull300Hz_CategoricalFocalLossFAIL.keras')

# Evaluate
h5_file = 'C:/Users/kaity/Documents/GitHub/Ecotype/MalahatBalanced_melSpec_8khz1.h5'


# Create the confusion matrix
val_batch_loader =  Eco.BatchLoader2(h5_file, 
                           trainTest = 'train', batch_size=1000,  n_classes=7,
                            minFreq=300)
confResults = Eco.confuseionMat(model= model, 
                                val_batch_loader=val_batch_loader)

