#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import tensorflow as tf
from tensorflow import keras

import os
import shutil


# In[2]:


# Check if directory exists
# if exists, return Model load from the directory and epoch trained
# else, return None and 0
def loadModel(directory):
    if not os.path.exists(directory):
        print('Directory \"' + directory + '\" doesn\'t exist. Check if \"' + directory + '_temp\" exists instead.')
        if not os.path.exists(directory + '_temp'):
            print('Directory \"' + directory + '_temp\" doesn\'t exist either, return None.')
            return None, 0
        else:
            print('Directory \"' + directory + '_temp\" exists!')
            print('****** WARNING ******  ' + directory + '_temp\" is used to load model because \"' + directory + '\" cannot be found.')
            directory = directory + '_temp'
    
    
    print('Loading model from \"' + directory + '\" ...')
    model = keras.models.load_model(directory)
    epochCount = 0
    if os.path.isfile(directory + '/epoch_count.txt'):
        with open(directory + '/epoch_count.txt', 'r') as f:
            epochCount = int(f.read())
    else:
        print('****** WARNING ******  Cannot find file \"' + directory + '/epoch_count.txt' + '\", return 0 as epoch count.')
    
    print('Model loaded.')
    return model, epochCount


# In[3]:


# First save the model into "directory_temp"
# After saving is completed, rename the folder to "directory"
def saveModel(model, directory, epochCount):
    # Remove directory_temp folder
    if os.path.exists(directory + '_temp'):
        shutil.rmtree(directory + '_temp')
    
    # Save the model to directory_temp
    print('Saving model as \"' + directory + '_temp\" ...')
    model.save(directory + '_temp')
    with open(directory + '_temp' + '/epoch_count.txt', 'w') as f:
        f.write(str(epochCount))
    print('Model saved.')
    
    # Remove directory folder is exists
    if os.path.exists(directory):
        shutil.rmtree(directory)
        
        
    # Rename directory_temp to directory after model saved
    print('Renaming \"' + directory + '_temp\" to \"' + directory + '\" ...')
    os.rename(directory + '_temp', directory)
    print('Finished.')

