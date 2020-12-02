#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

import numpy as np


# In[2]:


# Discriminator (Age discriminative network and Transition pattern discriminative network seems to be the same)
def createAgeDiscriminator():
    # Input image
    inputImage = keras.layers.Input((128, 128, 3))
    image1 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2))(inputImage)    #(63, 63, 64)
    
    # Input onehot age
    inputAge = keras.layers.Input((10,))    #(10,)
    age1 = keras.layers.Reshape((1, 1, 10))(inputAge)    #(1, 1, 11)
    age1 = keras.layers.UpSampling2D((128, 128))(age1)    #(128, 128, 10)
    age1 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2))(age1)    #(63, 63, 64)
    age1 = keras.layers.LeakyReLU()(age1)
    
    # Concate layer
    o = keras.layers.Concatenate()([image1, age1])    #(63, 63, 128)
    
    o = keras.layers.Conv2D(128, (4, 4), strides=(2, 2))(o)    #(30, 30, 128)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Conv2D(256, (4, 4), strides=(2, 2))(o)    #(14, 14, 256)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Conv2D(512, (4, 4), strides=(2, 2))(o)    #(6, 6, 512)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Conv2D(1024, (4, 4), strides=(2, 2))(o)    #(2, 2, 1024)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Conv2D(2, (2, 2), activation='sigmoid')(o)
    
    o = keras.layers.Reshape((2,))(o)
    
    model = keras.Model(inputs=[inputImage, inputAge], outputs=o)
    #keras.utils.plot_model(model, 'AgeDiscriminator.png', show_shapes=True, rankdir='TB', dpi=200)
    
    return model

