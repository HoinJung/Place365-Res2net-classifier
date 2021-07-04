from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import PIL
from PIL import Image
import PIL.Image as pilimg
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import math
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
'''
Making place list.
train dataset => './data/train'
test dataset => './data/test'

No need validation dataset since we use validation split from train dataset. 
You can setup the validation_split ratio in 'param_config.yml'

'''
path = "./data/"
place_list_1 = os.listdir(path+'train')
val_dir = path+'test/'

print("train place list : num = %i " % len(place_list_1))
print(place_list_1)

# create place label
encoder = LabelEncoder()

# training image list
train_img_list = []
train_place_list = []
for place in place_list_1 : 
    train_dir = os.path.join(path+'train/'+place)
    train_list = os.listdir(train_dir)
    for i in train_list : 
        train_imgs = os.path.join(train_dir+'/'+i)
        train_place_list.append(place)
        train_img_list.append(train_imgs)
train_img_df = pd.DataFrame(train_img_list)
train_img_df.columns=['image_path']
train_place_df=pd.DataFrame(train_place_list)
train_place_df.columns=['place']
place_label_train = np.array(train_place_df['place'])
encoder.fit(place_label_train)
train_class_df = pd.DataFrame(encoder.transform(place_label_train))
train_class_df.columns=['class']
train_df = pd.concat([train_img_df,  train_place_df, train_class_df],axis=1 )
train_df.to_csv(path+'train.txt',mode='w')
print("train place dataframe")
print(train_df.head())

print("completed")

# place_list_2 = os.listdir(path+'val')
place_list_2 = os.listdir(val_dir)

print("train place list : num = %i " % len(place_list_2))
print(place_list_2)

# test image list 작성
# train place list와 test list 다름.
test_img_list = []
test_place_list=[]
for place in place_list_2 : 
#     test_dir = os.path.join(path+'val/'+place)
    test_dir = os.path.join(val_dir+place)

    test_list = os.listdir(test_dir)
    
    for i in test_list : 
        test_imgs = os.path.join(test_dir+'/'+i)
        test_place_list.append(place)
        test_img_list.append(test_imgs)

test_img_df= pd.DataFrame(test_img_list)
test_img_df.columns=['image_path']
test_place_df=pd.DataFrame(test_place_list)
test_place_df.columns=['place']
place_label_test = np.array(test_place_df['place'])

# new label for only included in test set (unseen label)
for label in np.unique(place_label_test) :
    if label not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, label)
        
# encoder.fit(place_label)
# encoder.fit(place_label_train)
test_class_df = pd.DataFrame(encoder.transform(place_label_test))
test_class_df.columns=['class']
test_df = pd.concat([test_img_df,  test_place_df, test_class_df],axis=1 )
test_df.to_csv(path+'test.txt',mode='w')
print("test place dataframe")
print(test_df.head())
print(len(test_df))

print("completed")
