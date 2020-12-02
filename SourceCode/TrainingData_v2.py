import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow import keras

import UTKFaceLoader128_v2

import time
import matplotlib.pyplot as plt



# Generate a single face image
def generateFace(generator, inputImage, targetAgeGroup):
    i1 = np.expand_dims(inputImage, 0)
    i2 = np.expand_dims(UTKFaceLoader128_v2.ageGroupToOnehot(targetAgeGroup), 0)
    return generator([i1, i2])[0]


# Return a nparray of ageOnehot, image, real/fake label
# Each element is Either
# 1. A real image with corresponding age labeled as real (1.0, 0.0)
# Or
# 2. A fake image generated with an input image and age, labeled as fake (0.0, 1.0)
def realFakeImageTrainingPair(generator, masterDict, batchsize):
    real = int(batchsize * 0.5) ##################################
    fake = int(batchsize - real)
    
    # Return data
    ageOnehotSet = np.empty((batchsize, 10))
    imageSet = np.empty((batchsize, 128, 128, 3))
    labelSet = np.empty((batchsize, 2))
    
    counter = 0
    
    # Generate all real images
    i = 0
    while i < real:
        selectAgeGroup = np.random.randint(10) # Random select age group
        selectGender =  i % 2
        selectEthic = i % 5
        
        # Select random image from the list
        if len(masterDict[selectAgeGroup][selectEthic][selectGender]) <= 0:
            continue
        selectImage = np.random.randint(len(masterDict[selectAgeGroup][selectEthic][selectGender]))
        
        ageOnehotSet[counter] = UTKFaceLoader128_v2.ageGroupToOnehot(selectAgeGroup)
        imageSet[counter] = masterDict[selectAgeGroup][selectEthic][selectGender][selectImage]
        labelSet[counter] = np.array([1.0, 0.0])
        
        counter = counter + 1
        i = i + 1
        
    # Generate all fake images
    i = 0
    while i < fake:
        # Either image is fake/generated + random age
        # Or real image + incorrect age
        if np.random.randint(0, 9) < 5:
            # Use generated image
            
            selectAgeGroup = np.random.randint(10) # Random select age group
            selectGender = i % 2
            selectEthic = i % 5
        
            # Select random image from the list
            if len(masterDict[selectAgeGroup][selectEthic][selectGender]) <= 0:
                continue
            selectImage = np.random.randint(len(masterDict[selectAgeGroup][selectEthic][selectGender]))
            
            # Generate fake image based on input image + random age group
            inputImage = generateFace(generator, masterDict[selectAgeGroup][selectEthic][selectGender][selectImage], np.random.randint(10))
            
            # Generate random age group onehot
            inputAge = UTKFaceLoader128_v2.ageGroupToOnehot(np.random.randint(10))
            
            # -ve label
            inputLabel = np.array([0.0, 1.0])
            
        else:
            # Use real image with incorrect ageGroupOnehot
            
            selectAgeGroup = np.random.randint(10) # Random select age group
            selectGender = i % 2
            selectEthic = i % 5
        
            # Select random image from the list
            if len(masterDict[selectAgeGroup][selectEthic][selectGender]) <= 0:
                continue
            selectImage = np.random.randint(len(masterDict[selectAgeGroup][selectEthic][selectGender]))
            
            # Select ageGroupt != selectAgeGroup
            fakeAgeGroup = np.random.randint(10)
            while fakeAgeGroup == selectAgeGroup:
                fakeAgeGroup = np.random.randint(10)
            
            # Input real image
            inputImage = masterDict[selectAgeGroup][selectEthic][selectGender][selectImage]
            
            # Incorrect ageGroupOnehot
            inputAge = UTKFaceLoader128_v2.ageGroupToOnehot(fakeAgeGroup)
            
            # -ve label
            inputLabel = np.array([0.0, 1.0])
        
        imageSet[counter] = inputImage
        ageOnehotSet[counter] = inputAge
        labelSet[counter] = inputLabel
        
        counter = counter + 1
        i = i + 1
    
    return imageSet, ageOnehotSet, labelSet
        


# In[4]:


# Used to train generator
# Return nparray of image, ageOnehot, labels
# Image and ageOnehot are input to generator
# Labels are all positive for training the generator
def realImageTrainingPair(masterDict, batchsize):
    imageSet = np.empty((batchsize, 128, 128, 3))
    ageOnehotSet = np.empty((batchsize, 10))
    labelSet = np.empty((batchsize, 2))
    
    i = 0
    while i < batchsize:
        
        selectAgeGroup = np.random.randint(10) # Random select age group
        selectGender = i % 2
        selectEthic = i % 5
        
        # Select random image from the list
        if len(masterDict[selectAgeGroup][selectEthic][selectGender]) <= 0:
            continue
        selectImage = np.random.randint(len(masterDict[selectAgeGroup][selectEthic][selectGender]))
        
        # Image
        imageSet[i] = masterDict[selectAgeGroup][selectEthic][selectGender][selectImage]
        ageOnehotSet[i] = UTKFaceLoader128_v2.ageGroupToOnehot(selectAgeGroup)
        
        labelSet[i] = np.array([1.0, 0.0])    # Trick the discriminator to think it is real
        
        i = i + 1
        
    return imageSet, ageOnehotSet, labelSet

