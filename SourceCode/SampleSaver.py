#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os

import UTKFaceLoader128_v2


# In[2]:
# In[10]:


def saveSample(generator, outputDirectory, epochs, sampleDirectory = '../../Datasets/UTKSelectedSamples'):
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)
        
    if not os.path.exists(sampleDirectory):
        print('****** ERROR ****** \"' + sampleDirectory + '\" cannot found.')
        return
    
    # Sort the file
    tempList = os.listdir(sampleDirectory)
    for i in range(len(tempList)):
        temp = tempList[i].split('_')
        tempList[i] = int(temp[0]) * 10000 + int(temp[1]) * 10 + int(temp[2])
    posList = []
    for i in range(len(tempList)):
        posList.append(i)
    tempDict = dict(zip(tempList, posList))
    oList = os.listdir(sampleDirectory)
    imageList = os.listdir(sampleDirectory)
    tempList = sorted(tempList)
    for i in range(len(tempList)):
        imageList[i] = oList[tempDict[tempList[i]]]
    images = []
    for i in imageList:
        img = Image.open(sampleDirectory + '/' + i)
        img = img.resize((128, 128))
        img = np.array(img)
        img = img / 127.5 - 1.0
        images.append(img)
    ages = [0, 8, 15, 25, 35, 45, 55, 65, 75, 85]
    
    
    
    # Save as figure
    counter = 0
    y = len(imageList)
    x = 1 + len(ages)
    
    fig = plt.figure()
    
    for j in range(y):
        for i in range(x):
            counter = counter + 1
            plt.subplot(y, x, counter)
            plt.axis('off')
            
            if i == 0:
                if j == 0:
                    plt.title('Original', fontsize=6)
                plt.imshow(images[j] * 0.5 + 0.5)
            else:
                if j == 0:
                    plt.title(UTKFaceLoader128_v2.getAgeGroupLabel(ages[i - 1]), fontsize=6)
                # Generated image
                genImage = generator([np.expand_dims(images[j], 0), np.expand_dims(UTKFaceLoader128_v2.ageToOnehot(ages[i - 1]), 0)])[0]
                plt.imshow(genImage * 0.5 + 0.5)
    
    plt.savefig(outputDirectory + '/epoch_' + str(epochs) + '.jpg', dpi=300)
    plt.close()

