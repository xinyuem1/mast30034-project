# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:47:14 2020

@author: Windwalker
"""
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
#from task1_unetpp import unetpp
from task1_unet import unet
#from task1_resUnet import *
import pandas as pd

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
train_dir = './task1/train'
test_dir = './task1/test'

def myGenerator(base_dir):
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        base_dir,
        classes = ["image"],
        class_mode=None,
        color_mode="rgb",
        seed=seed,
        target_size=(128, 128))
    mask_generator = mask_datagen.flow_from_directory(
        base_dir,
        classes = ["label"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(128, 128),
        seed=seed)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        if (np.max(img) > 1):
            img = img / 255
            mask = mask /255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        yield (img,mask)
    
train_gen = myGenerator(train_dir)
test_gen = myGenerator(test_dir)

model = unet()
history =model.fit(train_gen, validation_data=test_gen,validation_steps=15,steps_per_epoch=64,epochs=10)

hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history_unet.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
model.save('model_unet.h5')

