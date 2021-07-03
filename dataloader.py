import os
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
import pandas as pd
import numpy as np



def data_generator(config, base_dir):
    
    # generate dataset splitting train_set and valid_st
    # call the parameters
    random_seed= 42
    shuffle_dataset = True   
    validation_split = 0.2
#     validation_split = config['data']['val_holdout_frac']
    batch_size = config['training']['batch_size']
    num_class = config['training']['num_class']
    
    # call Custom dataset
    train_dataset = CustomDataset(base_dir, num_class)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    
    #split and shuffle
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)        
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

        
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    
    return train_loader, valid_loader, len(valid_sampler)

def data_generator_test(config, base_dir):
    batch_size = config['training']['batch_size']
    num_class = config['training']['num_class']
    test_dataset = CustomDataset_test(base_dir, num_class)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    
    return test_loader, len(test_dataset)

class CustomDataset(Dataset):
    def __init__(self, base_dir, class_num):
        
        self.train_dir = base_dir + 'train/'
        self.train_data_list = base_dir + 'train.txt'
        
        self.class_num = class_num 
        #Onehot encoded manually
#         self.enc = OneHotEncoder()

        # open the csv(txt) file, and get data_path, label
        dataset=[]
        for line in open(self.train_data_list,'r'):
            spl = line.strip().split(',')
            dataset.append([spl[1],spl[3]])
            
        # remove index line
        del dataset[0]
        dataset_np = np.array(dataset)
        # input image root
        self.img_name = dataset_np[:,0]
        print("number of training data : %i" % len(self.img_name))
        
        # class if input images
        self.class_name = dataset_np[:,1].astype(np.int)
#         self.class_name2 = dataset_np[:,2].astype(np.int)

        
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
#         self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(90),transforms.RandomVerticalFlip(p=0.5)])
    def __len__(self):
        return len(self.img_name)
    
    
    def __getitem__(self, idx):
        
        # open input image
        img_name = self.img_name[idx]
        img_data=pilimg.open(img_name).resize((256,256))
        img_arr = np.array(img_data) / 255
        shape_val = img_arr.shape
        
        # some input images are monochrome photo. so unsqueeze to shape [256,256,3]
        if len(shape_val) == 3 :
            x_input = torch.FloatTensor( np.transpose(img_arr, (2, 0, 1)) )
        else :
            
            x_input = torch.FloatTensor( [img_arr, img_arr, img_arr] )
            
        # input tensor and class index
        x = torch.FloatTensor(x_input)
        x = self.transform(x)
        y =  self.class_name[idx]
        return x, y
#         y2 =  self.class_name2[idx]
#         return x, y1, y2



    
    
class CustomDataset_test(Dataset):
    def __init__(self, base_dir, class_num):
        
        # check directory to test. 'test' or 'val'
        self.test_dir = base_dir + 'test/'
        self.test_data_list = base_dir + 'test.txt'
        self.class_num = class_num 
        
        # same with train custom dataset
        dataset=[]
        class_str=[]
        for line in open(self.test_data_list,'r'):
            spl = line.strip().split(',')
            dataset.append([spl[1],spl[3]])
            class_str.append(spl[2])
        del dataset[0]
        del class_str[0]
        
        dataset_np = np.array(dataset)
     
        self.img_name = dataset_np[:,0]
        self.class_name = dataset_np[:,1].astype(np.int)
        
        # get ground truth class name as string.
        self.class_str = class_str[:]
    def __len__(self):
        return len(self.img_name)
    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img_data=pilimg.open(img_name).resize((256,256))
        
#         print('type : ', type(img_data))
#         print('shape1 :', type(pilimg.open(img_name)))
#         print('shape2 : ', type(img_data))
        img_arr = np.array(img_data) / 255
#         print(img_arr)
        shape_val = img_arr.shape

        if len(shape_val) == 3 :
            x_input = torch.FloatTensor( np.transpose(img_arr, (2, 0, 1)) )
            
        else :
            x_input = torch.FloatTensor( [img_arr, img_arr, img_arr] )
        
        x = torch.FloatTensor(x_input)
#         print(x.shape)
        y = self.class_name[idx]
        z = self.class_str[idx]
        return x, y,z,img_name