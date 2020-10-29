# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:46:56 2020

@author: Windwalker
"""

import os
import shutil
import random
input_dir = "../ISIC2018_Task1-2_Training_Input/"
target_dir = "../ISIC2018_Task1_Training_GroundTruth/"
img_paths = os.listdir(input_dir)

random.Random(2020).shuffle(img_paths)

splitPoint = int(len(img_paths)*0.8)

train_img = img_paths[:splitPoint]
test_img = img_paths[splitPoint:]
len(train_img)

train_label = list(map((lambda x:x.split('.')[0]+"_segmentation.png"),train_img))
test_label = list(map((lambda x:x.split('.')[0]+"_segmentation.png"),test_img))

for fname in train_img:
    src = os.path.join(input_dir,fname)
    dst = os.path.join("./train/image",fname)
    shutil.copyfile(src, dst)
    
for fname in test_img:
    src = os.path.join(input_dir,fname)
    dst = os.path.join("./test/image",fname)
    shutil.copyfile(src, dst)

for fname in train_label:
    src = os.path.join(target_dir,fname)
    dst = os.path.join("./train/label",fname.split("_segment")[0]+".png")
    shutil.copyfile(src, dst)
    
for fname in test_label:
    src = os.path.join(target_dir,fname)
    dst = os.path.join("./test/label",fname.split("_segment")[0]+".png")
    shutil.copyfile(src, dst)