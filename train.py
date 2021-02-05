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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.model_name == 'Res2Net' :
            self.model = Res2Net().to(self.device)
        else : 
            print("Wrong Model Name")
        self.model_path = self.config['result_path']
        self.num_class = self.config['training']['num_class']
        self.data_dir = self.config['data']['data_dir']
        self.train_datagen, self.val_datagen = data_generator(self.config, self.data_dir) 
        
        self.loss = nn.CrossEntropyLoss()
        self.lr = self.config['training']['learing_rate']
        self.opt = self.config['training']['optimizer']
        if self.opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else : 
            print("Wrong optimzer name, only Adam")

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,milestones=[50,100], gamma=0.5)
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
        ## training Start
        for epoch in range(self.epochs):
            self.scheduler.step()
            self.model.train()
            print("Beginning training epch {}".format(epoch+1))
            check_lr = get_lr(self.optimizer)
            
            for batch_idx, samples in enumerate(self.train_datagen):
                x_train, y_train = samples
                data, target = x_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(input=output,target=target)
                loss.backward()
                self.optimizer.step()

                train_loss_check.append(loss.item())
                if( batch_idx % 20 == 0 ):
                    print('    loss at batch {}: {}'.format(batch_idx, loss), flush=True)


            
            with torch.no_grad(): # very very very very important!!!
                self.model.eval()
                torch.cuda.empty_cache()
                val_loss = []
                correct = 0

                for batch_idx, samples in enumerate(self.val_datagen):
                    x_test, y_test = samples
                    data, target = x_test.to(self.device), y_test.to(self.device)
                    output= self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)          
#                     target_to_cal = target.argmax(dim=1, keepdim=True)
#                     correct += pred.eq(target_to_cal.view_as(pred)  ).sum().item()
                    

                    # loss value check
                loss_val = self.loss(input=output,target=target)
                val_loss.append(loss_val)
                self.scheduler.step()
#                 corr_val = (100. * correct) / len(self.val_datagen)
                
            val_loss = torch.mean(torch.stack(val_loss))
            print("validation loss : {:.6f}".format( val_loss) )
#                 print("validation accuracy : {:.6f}".format(corr_val) )
            check_continue = TorchEarlyStopping( loss.detach().cpu().numpy(),
                    val_loss.detach().cpu().numpy())
            if not check_continue:
                break
            val_loss_set.append(val_loss)
            PATH_ckpt = self.model_path + "epoch_" + str(epoch).zfill(3) + '_ckpt.pth'
            torch.save(self.model.state_dict(), PATH_ckpt)
            print("")
            print("")
        self.save_model()
        temp_file_name = self.model_path+"loss_val_save.csv"
        np.savetxt(temp_file_name, val_loss_set, delimiter=",")
        print("training completed")
         
        return True
        
    def save_model(self):
        """Save the final model output."""
        if self.framework == 'keras':
            self.model.save(self.config['training']['model_dest_path'])
        elif self.framework == 'torch':
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(), self.model_path+'final.pth')
            else:
                torch.save(self.model.state_dict(), self.model_path+'final.pth')

"""to print learning schedule"""
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
    
""" from here to end. for early stopping !!"""    
class TorchEarlyStopping(object):

    def __init__(self, patience=5, threshold=0.0, verbose=False):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, metric_score):

        if self.best is None:
            self.best = metric_score
            self.counter = 0
        else:
            if self.best - self.threshold < metric_score:
                self.counter += 1
            else:
                self.best = metric_score
                self.counter = 0

        if self.counter >= self.patience:
            self.stop = True
