# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 06:03:21 2020

@author: Windwalker
"""

from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,UpSampling2D,concatenate,BatchNormalization
from keras.optimizers import Adam

def unetpp(pretrained_weights = None,input_size = (128,128,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    up2 = Conv2D(32, 2, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
    merge2 = concatenate([conv1,up2],axis=3)
    conv3 = Conv2D(32, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv3 = Conv2D(32, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(conv3)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv4 = Conv2D(128, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv4 = Conv2D(128, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    up4 = Conv2D(64, 2, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge4 = concatenate([conv2,up4],axis=3)
    conv5 = Conv2D(64, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv5 = Conv2D(64, 3,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up5 = Conv2D(32, 2,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    
    merge5 = concatenate([conv1,conv3,up5],axis=3)
    conv6 = Conv2D(32,3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv6 = Conv2D(32,3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2D(2,3, activation="relu",padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv7 = Conv2D(1,1, activation="sigmoid")(conv6)
    model = Model(input = inputs, output = conv7)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model