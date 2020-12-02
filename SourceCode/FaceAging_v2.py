#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import os
import sys

import AgeDiscriminator
import Generator
import ModelLoader
import UTKFaceLoader128_v2

import SampleSaver
import TrainingData_v2
import EpochHistorySaver

print('Tensorflow version', tf.__version__)
print('Numpy version', np.__version__)

# Allow memory growth
physicalDevices = tf.config.experimental.list_physical_devices('GPU')
for i in physicalDevices:
    tf.config.experimental.set_memory_growth(i, True)


# In[2]:


masterDict = UTKFaceLoader128_v2.load()


# In[3]:


print('Old size', len(masterDict[0][0][0]))


# In[4]:


UTKFaceLoader128_v2.augmentData(masterDict, 0.7)


# In[5]:


print('New size', len(masterDict[0][0][0]))


# In[6]:


#selectAge = UTKFaceLoader128_v2.ageToAgeGroup(np.random.randint(100))
#selectGender = np.random.randint(2)
#imageList = masterDict[selectAge][selectGender]
#selectImage = np.random.randint(len(imageList))
#print('AgeGroup', selectAge, 'Gender', selectGender)
#plt.imshow(imageList[selectImage] / 2.0 + 0.5)


# In[7]:


epochCounter = 0


# In[8]:


# Create generator
generator, epochCounter = ModelLoader.loadModel('Generator')
if generator == None:
    print('Generator is not saved before, so create new instead')
    generator = Generator.createGenerator()
    generator.compile(optimizer='adam', loss='mean_squared_error')
    
# Set lower learning rate?


# In[9]:


# Create age discriminator
ageDiscriminator, epochCounter = ModelLoader.loadModel('AgeDiscriminator')
if ageDiscriminator == None:
    print('AgeDiscriminator is not saved before, so create new instead')
    ageDiscriminator = AgeDiscriminator.createAgeDiscriminator()
    ageDiscriminator.compile(optimizer='adam', loss='categorical_crossentropy')
    
# Set lower learning rate?


# In[10]:


# Combine generator and ageDiscriminator to train generator
inputImage = keras.layers.Input((128, 128, 3))
inputAge = keras.layers.Input((10,))

generator.trainable = True
ageDiscriminator.trainable = False

o = generator([inputImage, inputAge])
o = ageDiscriminator([o, inputAge])

ageGAN = keras.Model(inputs=[inputImage, inputAge], outputs=o)
ageGAN.compile(optimizer='adam', 'categorical_crossentropy')


ageDiscriminator.trainable = True


# In[11]:


# Generate a single face image
def generateFace(inputImage, targetAgeGroup):
    i1 = np.expand_dims(inputImage, 0)
    i2 = np.expand_dims(UTKFaceLoader128_v2.ageGroupToOnehot(targetAgeGroup), 0)
    return generator([i1, i2])[0]


# In[12]:


adLoss = []
ganLoss = []
_, adLoss, ganLoss = EpochHistorySaver.loadEpochHistory('LossHistory.txt')
if adLoss == None:
    epochCounter = 0
    adLoss = []
    ganLoss = []


# In[13]:


# Training the network
def trainOnce(batchSize):
    
    # Get data to train ageDiscriminator
    adImage, adAge, adLabel = TrainingData_v2.realFakeImageTrainingPair(generator, masterDict, batchSize)
    
    # Train ageDis...
    newAdLoss = ageDiscriminator.train_on_batch((adImage, adAge), adLabel)
    
    # Get data to train GAN
    ganImage, ganAge, ganLabel = TrainingData_v2.realImageTrainingPair(masterDict, batchSize)
    
    # Trai GAN...
    newGanLoss = ageGAN.train_on_batch((ganImage, ganAge), ganLabel)
    
    return newAdLoss, newGanLoss


# In[14]:


movingAvg = 10.0
# If movingAvg == 0.0, something bad happened


# In[ ]:


for i in range(100000000):
    epochCounter = epochCounter + 1
    newAdLoss, newGanLoss = trainOnce(50)
    print('Epoch %5d, AdLoss: %5.5f, GANLoss: %5.5f                   ' % (epochCounter, newAdLoss, newGanLoss), end='\r')
    adLoss.append(newAdLoss)
    ganLoss.append(newGanLoss)
    
    if epochCounter % 100 == 0:
        SampleSaver.saveSample(generator, 'Output', epochCounter)
        
    if epochCounter % 1000 == 0:
        # Save model and save loss history
        ModelLoader.saveModel(generator, 'Generator', epochCounter)
        ModelLoader.saveModel(ageDiscriminator, 'AgeDiscriminator', epochCounter)
        EpochHistorySaver.saveEpochHistory('LossHistory.txt', adLoss, ganLoss, 1)
    
    movingAvg = movingAvg * 0.5 + abs(newAdLoss) * 0.5 + abs(newGanLoss) * 0.5
    if abs(movingAvg) < 0.001:
        print('Both losses are zero, sth bad happened.')
        sys.exit('Both losses became zero!!!')

