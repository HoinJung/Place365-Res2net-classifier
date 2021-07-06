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

import time
from model import Res2Net
from dataloader import data_generator
now = time.localtime()

class Trainer(object):

    def __init__(self, config):
        self.config = config
        
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epoch']
        self.model_name = self.config['model_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_name == 'Res2Net' :
            self.model = Res2Net(self.config).to(self.device)
            self.model = torch.nn.DataParallel(self.model)
        else : 
            print("Wrong Model Name")
        self.model_path = self.config['result_path']
        self.num_class = self.config['training']['num_class']
        self.data_dir = self.config['data']['data_dir']
        self.train_datagen, self.val_datagen, self.num = data_generator(self.config, self.data_dir) 
        
        self.loss = nn.CrossEntropyLoss()
        self.patience = self.config['training']['callbacks']['early_stopping']['patience']

        self.lr = self.config['training']['learing_rate']
        self.opt = self.config['training']['optimizer']
        self.opt = self.config['training']['optimizer']
        if self.opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else : 
            print("Wrong optimzer name, only Adam")

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,milestones=[50,100,150,200,250], gamma=0.5)
        self.framework = 'torch'

        
        
    def train(self):       
        print()
        print("=======================")
        print("Trainging Start")
        print("model : ", self.model_name)
        print("batch size : ",self.batch_size)
        print("epoch : ", self.epochs)
        print("loss : ",self.loss)
        print("starting time : %04d/%02d/%02d %02d:%02d:%02d"% (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
        print("=======================")
        print()
        train_loss_check = []
        loss_set = []
        
        val_loss_set = []
        stop = False
        best = None
        
        
        ## training Start
        self.model.train()
        for epoch in range(self.epochs):
            self.scheduler.step()
            
            print("Beginning training epch {}".format(epoch+1))
            check_lr = get_lr(self.optimizer)
            
            for batch_idx, samples in enumerate(self.train_datagen):
                x_train, y_train = samples
                data, target = x_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.loss(output,target)


                loss.backward()
                self.optimizer.step()

                if( batch_idx % 100 == 0 ):

                    
                    print('    loss at batch {}: {}'.format(batch_idx, loss), flush=True)


            self.model.eval()
            
            with torch.no_grad(): 
                
                #torch.cuda.empty_cache()
                val_loss = []
                correct_1 = 0
                correct , avg_loss= 0,0
                cnt = 0
                print('Validation...')
                for batch_idx, samples in enumerate(self.val_datagen):
                    cnt+=1
                    x_test, y_test = samples
                    data, target = x_test.to(self.device), y_test.to(self.device)
                    output= self.model(data)
                    loss_val = self.loss(output,target)

                    val_loss.append(loss_val.item())

                    output = F.softmax(output,dim=0)
                    pred_1 = output.argmax(dim=1, keepdim=True)
                    crr = pred_1.eq( target.view_as(pred_1)  ).sum().item()

                    correct_1 += crr
                  
                    
            corr_1_val = (100. * correct_1) / self.num

            
            print("validation loss : {:.6f}".format( np.mean(val_loss) ))
            print("validation accuracy : %.2f" % corr_1_val) 
            self.scheduler.step()
        
            PATH_ckpt = self.model_path + "epoch_" + str(epoch+1).zfill(3) + '_ckpt.pth'
            patience = self.patience
            
            val_loss = np.mean(val_loss)
            if best is None :
                best = val_loss 
                counter = 0
            else : 
                if best < val_loss:
                    counter += 1
                else :
                    best = val_loss
                    self.save_model(PATH_ckpt)
                    counter = 0
                    print("____model saved at epoch : {0:d}, validation loss : {1:.2f}".format(epoch,val_loss))
            if counter >= patience :
                stop = True
            if stop:
                print("EarlyStopping Trigger")
                break
            val_loss_set.append(val_loss)
            print("")
            print("")
        self.save_model(self.model_path+'final.pth')
        temp_file_name = self.model_path+"loss_val_save.csv"
        np.savetxt(temp_file_name, val_loss_set, delimiter=",")
        print("training completed")
         
        return True
    
    
    def save_model(self,pth):
        """Save the final model output."""
        if self.framework == 'keras':
            self.model.save(self.config['training']['model_dest_path'])
        elif self.framework == 'torch':
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(),pth)
            else:
                torch.save(self.model.state_dict(), pth)
        

"""to print learning schedule"""
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    