#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

import numpy as np


# In[2]:


# Generator
def createGenerator():
    # Input image of original person
    inputImage = keras.layers.Input(shape=(128, 128, 3))    #(128, 128, 3)
    
    #Input label of target age
    inputAge = keras.layers.Input(shape=(10,))    #(10,)
    
    # Combine input image and age into 1 tensor
    age1 = keras.layers.Reshape((1, 1, 10))(inputAge)    #(1, 1, 10)
    age1 = keras.layers.UpSampling2D((128, 128))(age1)    #(128, 128, 10)
    imageAge = keras.layers.Concatenate()([inputImage, age1])
    
    
    # First section
    o = keras.layers.Conv2D(3, (3, 3), padding='same')(imageAge)    #(128, 128, 3)
    s = keras.layers.LeakyReLU()(o)    #(128, 128, 3), used as skip layer
    o = keras.layers.BatchNormalization()(s)
    
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)    #(128, 128, 64)
    o = keras.layers.LeakyReLU()(o)
    o = keras.layers.BatchNormalization()(o)
    
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)    #(128, 128, 64)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Concatenate()([o, s])
    
    
    # Second section
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (1, 1), padding='same')(o)
    s = keras.layers.LeakyReLU()(o)    #Used as skip layer
    
    o = keras.layers.BatchNormalization()(s)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Concatenate()([o, s])
    
    
    # Third section
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (1, 1), padding='same')(o)
    ss = keras.layers.LeakyReLU()(o)    #Used as super skip layer
    
    o = keras.layers.BatchNormalization()(ss)
    o = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(o)
    s = keras.layers.LeakyReLU()(o)    #Used as skip layer
    
    o = keras.layers.BatchNormalization()(s)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)    #Used as skip layer
    
    o = keras.layers.Concatenate()([o, s])
    
    
    # Fourth section(same as 2nd section)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (1, 1), padding='same')(o)
    s = keras.layers.LeakyReLU()(o)    #Used as skip layer
    
    o = keras.layers.BatchNormalization()(s)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Concatenate()([o, s])    
    
    
    # Fifth section(same as 2nd section)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (1, 1), padding='same')(o)
    s = keras.layers.LeakyReLU()(o)    #Used as skip layer
    
    o = keras.layers.BatchNormalization()(s)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Concatenate()([o, s])
    
    
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(64, (1, 1), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(3, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    o = keras.layers.Concatenate()([o, ss])
    
    
    # Sixth section
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(3, (1, 1), padding='same')(o)
    s = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(s)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(3, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Concatenate()([o, s])
    
    
    # Seventh section(same as 6th section)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(3, (1, 1), padding='same')(o)
    s = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(s)
    o = keras.layers.Conv2D(64, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Conv2D(3, (3, 3), padding='same')(o)
    o = keras.layers.LeakyReLU()(o)
    
    o = keras.layers.Concatenate()([o, s])
    
    
    o = keras.layers.Conv2D(3, (1, 1), padding='same')(o)
    o = keras.layers.Concatenate()([o, inputImage])
    o = keras.layers.Conv2D(3, (1, 1), padding='same', activation='tanh')(o)
    

    model = keras.Model(inputs=[inputImage, inputAge], outputs=o)
    #keras.utils.plot_model(model, 'Generator.png', show_shapes=True, rankdir='TB', dpi=200)
    
    return model

