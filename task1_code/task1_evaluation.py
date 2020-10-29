# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 03:33:20 2020

@author: Windwalker
"""
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt 

from PIL import Image
img = Image.open('./test/image/ISIC_0000042.jpg')
img = img.resize((128,128), Image.ANTIALIAS)
plt.imshow(img)
plt.show()

# Read Images 
mask = Image.open('./test/label/ISIC_0000042.png') 
# Output Images 
mask = mask.resize((128,128), Image.ANTIALIAS)
plt.imshow(mask,cmap="gray")
plt.show()

model1 = load_model("my_res_unetpp.h5")
pred1 = model1.predict(np.expand_dims(np.array(img)/255,axis=0))
pred1[pred1>0.5]=1
pred1[pred1<=0.5]=0
plt.imshow((np.array(img)*np.repeat(pred1[0],3,axis=2)).astype(np.uint8))
plt.show()

model2 = load_model("model_unetpp.h5")
pred2 = model2.predict(np.expand_dims(np.array(img)/255,axis=0))
pred2[pred2>0.5]=1
pred2[pred2<=0.5]=0
plt.imshow(pred2[0],cmap="gray",vmin=0,vmax=1)
plt.show()

model3 = load_model("model_unet.h5")
pred3 = model3.predict(np.expand_dims(np.array(img),axis=0))
pred3[pred3>0.5]=1
pred3[pred3<=0.5]=0
plt.imshow(pred3[0],cmap="gray",vmin=0,vmax=1)
plt.show()


unet_history = pd.read_csv("history_unet.csv")
plt.plot(unet_history['accuracy'])
plt.plot(unet_history['val_accuracy'])
plt.title('Model Accuracy of Unet')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0,8)
plt.ylim(0.75,0.95)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

unetpp_history = pd.read_csv("history_unetpp.csv")
plt.plot(unetpp_history['accuracy'])
plt.plot(unetpp_history['val_accuracy'])
plt.title('Model Accuracy of Unet++')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0,7)
plt.ylim(0.75,0.95)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

resunetpp_history = pd.read_csv("history_resunetpp.csv")
plt.plot(resunetpp_history['accuracy'])
plt.plot(resunetpp_history['val_accuracy'])
plt.title('Model Accuracy of Unet++(with identity blocks)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0,8)
plt.ylim(0.75,0.95)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

unet_history = pd.read_csv("history_unet.csv")
plt.plot(unet_history['loss'])
plt.plot(unet_history['val_loss'])
plt.title('Binary cross entropy of Unet')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.xlim(0,8)
plt.ylim(0.1,0.6)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

unetpp_history = pd.read_csv("history_unetpp.csv")
plt.plot(unetpp_history['loss'])
plt.plot(unetpp_history['val_loss'])
plt.title('Binary cross entropy of Unet++')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.xlim(0,7)
plt.ylim(0.1,0.6)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

resunetpp_history = pd.read_csv("history_resunetpp.csv")
plt.plot(resunetpp_history['loss'])
plt.plot(resunetpp_history['val_loss'])
plt.title('Binary cross entropy of Unet++(with identity blocks)')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.xlim(0,8)
plt.ylim(0.1,0.6)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
