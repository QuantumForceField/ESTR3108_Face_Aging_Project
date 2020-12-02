#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # UTK Face Loader
# ***
# ### Return 4 items
# - Name of the image (string)
# - Age (integer) [1-116]
# - AgeOnehot nparray (0-10, 10-20, ... 80-90, 90+), divided into 10 classes
# - Gender (integer) (0 = Male, 1 = Female)
# - Ethic (integer) (0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = others)
# - Image nparray (128 x 128 x 3) normalized [-1.0, 1.0]
# ***

import numpy as np
import os
from PIL import Image


import time
import matplotlib.pyplot as plt
import numpy as np

# Age grouping
# <5, <10, <18, <26, <36, <46, <58, <69, <80, >= 80

# Translate age to onehot
def ageToOnehot(age):
    if age < 5:
        ageOnehot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif age < 10:
        ageOnehot = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif age < 18:
        ageOnehot = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif age < 26:
        ageOnehot = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif age < 36:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif age < 46:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    elif age < 58:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif age < 69:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    elif age < 80:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    else:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    return ageOnehot

# Translate ageGroup to onehot
def ageGroupToOnehot(ageGroup):
    if ageGroup == 0:
        ageOnehot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif ageGroup == 1:
        ageOnehot = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif ageGroup == 2:
        ageOnehot = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif ageGroup == 3:
        ageOnehot = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif ageGroup == 4:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif ageGroup == 5:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    elif ageGroup == 6:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif ageGroup == 7:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    elif ageGroup == 8:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    elif ageGroup == 9:
        ageOnehot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    return ageOnehot

# Translate age to age group starting from 1
def ageToAgeGroup(age):
    if age < 5:
        return 0
    elif age < 10:
        return 1
    elif age < 18:
        return 2
    elif age < 26:
        return 3
    elif age < 36:
        return 4
    elif age < 46:
        return 5
    elif age < 58:
        return 6
    elif age < 69:
        return 7
    elif age < 80:
        return 8
    else:
        return 9

    

def getAgeGroupLabel(age):
    ageGroup = ageToAgeGroup(age)
    if ageGroup == 0:
        return '0-5'
    elif ageGroup == 1:
        return '5-10'
    elif ageGroup == 2:
        return '10-18'
    elif ageGroup == 3:
        return '18-26'
    elif ageGroup == 4:
        return '26-36'
    elif ageGroup == 5:
        return '36-46'
    elif ageGroup == 6:
        return '46-58'
    elif ageGroup == 7:
        return '58-69'
    elif ageGroup == 8:
        return '69-80'
    elif ageGroup == 9:
        return '80+'
    else:
        return 'Error'
    
def augmentData(masterDict, rate=0.3):
    # Augment data by fliping image horizontally
    
    # For each ageGroup
    for a in range(10):
        
        # For each ethic
        for e in range(4):
        
            # For each gender
            for g in range(2):
            
                origionalCount = len(masterDict[a][e][g])
            
                for i in range(origionalCount):
                
                    # Do augmentation by chance
                    if np.random.uniform() < rate:
                    
                        newImage = np.flip(masterDict[a][e][g][i], 1)
                        masterDict[a][e][g].append(newImage)
    

def load(directory = '../../Datasets/UTKFaceCleared'):
    # Get file list
    file = os.listdir(directory)
    print('There are total', len(file), 'files in \"' + str(directory) + '\"')
    
    # Create dictionary, each element is a ethicDict{ 0-4 : genderDict{ 0-1 : [image] } }
    masterDict = {}
    for i in range(10):
        # Create dictionary for each ethic
        ethicDict = {}
        
        for j in range(5):
            # Create dictionary for each gender
            genderDict = {0 : [], 1 : []}
            ethicDict[j] = genderDict
            
        masterDict[i] = ethicDict
    
    counter = 0
    total = len(file)
    errorFile = 0
    loadedFile = 0
    
    for f in file:
        # Load data percentage counter
        if(counter % 333 == 0):
            print('Processing %d outof %d files, %.1f %% loaded...' % (counter, total, 100.0 * counter / total), end = '\r')
        counter = counter + 1
        
        a_g_e = f.split('_')
        if(len(a_g_e) != 4):
            print('File \"' + directory + '/' + f + '\" discarded.                     ')
            errorFile = errorFile + 1
            continue
        
        name = f
        ageGroup = ageToAgeGroup(int(a_g_e[0]))
        gender = int(a_g_e[1])
        ethic = int(a_g_e[2])
        image = np.array(Image.open(directory + '/' + f).resize((128, 128)))
        # Normalize image
        image = image / 127.5 - 1.0
        
        # Append to master dict
        masterDict[ageGroup][ethic][gender].append(image)
        
        
        loadedFile = loadedFile + 1
        
    
    print('Finished loading data. %d loaded, %d discarded. Please have fun!' % (loadedFile, errorFile))
    return masterDict


# In[2]:


