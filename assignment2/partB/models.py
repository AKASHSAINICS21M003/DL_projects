#! /usr/bin/env python3

## Libraries: 
#### tensorflow version: 2.6.0

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from tensorflow.keras import  models, optimizers, layers, activations
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D ,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization


class inc_v2(object):
    """
    Inception V2
    """
    def __init__(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.input_size =input_size
        self.output_size=output_size
        self.drop_out=drop_out
        self.batch_normalization=batch_normalization
        self.initialize(input_size,output_size,drop_out,batch_normalization,dense_size)
    def initialize(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.model_b=InceptionResNetV2(include_top=False,
            weights="imagenet",
            input_shape=input_size,
        )
        self.model_b.trainable=False
        input = keras.Input(shape=input_size)
        x = self.model_b(input, training=False)
        self.model=Sequential()
        self.model.add(Model(input, x))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size))
        self.model.add(Activation('selu'))
        if batch_normalization==True:
            self.model.add(BatchNormalization())
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(output_size))
        self.model.add(Activation("softmax"))


class inc_v3(object):
    """
    Inception V3
    """
    def __init__(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.input_size =input_size
        self.output_size=output_size
        self.drop_out=drop_out
        self.batch_normalization=batch_normalization
        self.initialize(input_size,output_size,drop_out,batch_normalization,dense_size)
    def initialize(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.model_b=InceptionV3(include_top=False,
            weights="imagenet", 
            input_shape=input_size,
        )
        self.model_b.trainable=False
        input = keras.Input(shape=input_size)
        x = self.model_b(input, training=False)
        self.model=Sequential()
        self.model.add(Model(input, x))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size))
        self.model.add(Activation('selu'))
        if batch_normalization==True:
            self.model.add(BatchNormalization())
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(output_size))
        self.model.add(Activation("softmax"))

class xception(object):
    """
    Xception
    """
    def __init__(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.input_size =input_size
        self.output_size=output_size
        self.drop_out=drop_out
        self.batch_normalization=batch_normalization
        self.initialize(input_size,output_size,drop_out,batch_normalization,dense_size)
    def initialize(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.model_b=Xception(include_top=False,
            weights="imagenet",
            input_shape=input_size)
        self.model_b.trainable=False
        input = keras.Input(shape=input_size)
        x = self.model_b(input, training=False)
        self.model=Sequential()
        self.model.add(Model(input, x))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size))
        self.model.add(Activation('selu'))
        if batch_normalization==True:
            self.model.add(BatchNormalization())
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(output_size))
        self.model.add(Activation("softmax"))


class resnet(object):
    """
    Resnet
    """
    def __init__(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.input_size =input_size
        self.output_size=output_size
        self.drop_out=drop_out
        self.batch_normalization=batch_normalization
        self.initialize(input_size,output_size,drop_out,batch_normalization,dense_size)
    def initialize(self,input_size,output_size,drop_out,batch_normalization,dense_size):
        self.model_b=ResNet50(include_top=False,
            weights="imagenet",
            input_shape=input_size)
        self.model_b.trainable=False
        input = keras.Input(shape=input_size)
        x = self.model_b(input, training=False)
        self.model=Sequential()
        self.model.add(Model(input, x))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size))
        self.model.add(Activation('selu'))
        if batch_normalization==True:
            self.model.add(BatchNormalization())
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(output_size))
        self.model.add(Activation("softmax"))


