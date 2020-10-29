# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:35:32 2020

@author: Windwalker， Ruichen Kang
"""

from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,UpSampling2D,concatenate,BatchNormalization,Add,Activation
from keras.optimizers import Adam

def identity_block(X, filters):
    f1, f2, f3 = filters
    copy = Conv2D(f3,1, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(X)
    
    # first two steps
    X = Conv2D(f1, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization()(X)
    X = Conv2D(f2, 3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization()(X)
    
    # addition step
    X = Conv2D(f3, 3, padding = 'same', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization()(X)
    print(X.shape,copy.shape)
    X = Add()([X, copy])
    X = Activation('relu')(X)
    
    return X

def res_unetpp(pretrained_weights = None,input_size = (128,128,3)):
    inputs = Input(input_size)
    conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    res1 = identity_block(conv0,[32,32,32])
    conv1 = BatchNormalization()(res1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    res2 = identity_block(pool1,[64,64,64])
    conv2 = BatchNormalization()(res2)
    up2 = Conv2D(32, 2, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
    merge2 = concatenate([conv1,up2],axis=3)
    
    res3 = identity_block(merge2,[32,32,32])
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    res4 = identity_block(pool2,[128,128,128])
    
    up4 = Conv2D(64, 2, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res4))
    merge4 = concatenate([conv2,up4],axis=3)
    
    res5 = identity_block(merge4,[64,64,64])
    
    up5 = Conv2D(32, 2,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res5))
    
    merge5 = concatenate([conv1,res3,up5],axis=3)
    
    res6 = identity_block(merge5,[32,32,32])
    conv6 = Conv2D(32,3, activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(res6)
    conv6 = Conv2D(2,3, activation="relu",padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv7 = Conv2D(1,1, activation="sigmoid")(conv6)
    model = Model(input = inputs, output = conv7)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model