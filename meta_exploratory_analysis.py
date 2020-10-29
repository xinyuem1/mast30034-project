# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 00:20:15 2020

@author: Windwalker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import random

df = pd.read_csv("./HAM10000_metadata.csv")
dx_dict = {
    'nv': 'Melanocytic nevus',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like \nlesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratosis / \nBowen’s disease',
    'vasc': 'Vascular lesion',
    'df': 'Dermatofibroma'
}

dx_type_dict ={
    "histo" : "Histopathology",
    "confocal" :"Confocal",
    "follow_up":"Follow-up",
    "consensus":"Consensus"
}

# Missing Value
df.isnull().sum()
df['age'].fillna((df['age'].mean()), inplace=True)

# Label Distribution
df["label"] = df.dx.map(dx_dict.get)
label_count= df["label"].value_counts()
plt.barh(label_count.index,label_count.values,color="green")
plt.ylabel("Disease States",fontsize=16)
plt.xlabel("Number of Records",fontsize=16)
plt.title("Distribution of Disease States",fontsize=23)
plt.show()

# Sex
df["label"] = df.dx.map(dx_dict.get)
man = df[df["sex"]=="male"]
female = df[df["sex"]=="female"]
man_count = man["label"].value_counts()
female_count = female["label"].value_counts()
N = 7
height = 0.35
ind = np.arange(N)
fig, ax = plt.subplots()
p1 = ax.barh(ind, man_count.values, height, left=0)
p2 = ax.barh(ind+ height,female_count.values, height, left=0)
ax.set_title('Disease States by gender',fontsize=20)
ax.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.xlabel("Number of Records",fontsize=14)
plt.yticks(ind,man_count.index)
plt.show()

# Feature Distribution (Technical Validation field)
df["dx_type"] = df["dx_type"].map(dx_type_dict.get)
dx_type_count= df["dx_type"].value_counts()
plt.bar(dx_type_count.index,dx_type_count.values,color="orange")
plt.xlabel("Technical Validation method",fontsize=14)
plt.ylabel("Number of Records",fontsize=14)
plt.title("Distribution of Technical Validation method",fontsize=20)
plt.show()

# Feature Distribution (Localization)
loc_count= df["localization"].value_counts()
plt.barh(loc_count.index, loc_count.values,color="blue")
plt.ylabel("Localization",fontsize=14)
plt.xlabel("Number of Records",fontsize=14)
plt.title("The Distribution of Localization field",fontsize=20)
plt.show()

# Feature Distribution (age)
fig=sns.displot(df.age)
plt.xlabel("Age",fontsize=14)
plt.ylabel("Density",fontsize=14)
plt.title("Distribution of Age", fontsize=20)
plt.show(fig)

# Distribution of Disease Status by Age
fig=sns.displot(df,x="age",hue="label",kind="kde", fill=True)
plt.xlim(0,100)
plt.xlabel("Age",fontsize=14)
plt.ylabel("Density",fontsize=14)
plt.title("Distribution of Disease Status by Age",fontsize=20)
plt.show(fig)

"""

# Images
fig, m_axs = plt.subplots(7, 5, figsize = (4*5, 3*7))
train_path = './base_dir/train_dir'
folders = os.listdir(train_path)
for label,n_axs in zip(folders,m_axs):
    folder = os.path.join(train_path,label)
    imgs = random.sample(os.listdir(folder), 5)
    n_axs[0].set_title(dx_dict.get(label))
    for img,c_ax in zip(imgs,n_axs):
        c_ax.imshow(Image.open(os.path.join(folder,img)))
        c_ax.axis('off')
        
"""

        