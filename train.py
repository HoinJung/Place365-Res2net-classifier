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
def cross_entropy(pred, target, size_average=True):
 
    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(pred), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(pred), dim=1))
    
criterion = cross_entropy
class Trainer(object):

    def __init__(self, config):
        self.config = config
        
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epoch']
        self.model_name = self.config['model_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.device = self.model.cuda()
        if self.model_name == 'Res2Net' :
            self.model = Res2Net().to(self.device)
            self.model = torch.nn.DataParallel(self.model)
        else : 
            print("Wrong Model Name")
        self.model_path = self.config['result_path']
        self.num_class = self.config['training']['num_class']
        self.data_dir = self.config['data']['data_dir']
        self.train_datagen, self.val_datagen, self.num = data_generator(self.config, self.data_dir) 
        
        self.loss = nn.CrossEntropyLoss()
#         self.loss = CRI()
#         self.loss = FocalLoss()
        self.lr = self.config['training']['learing_rate']
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
#                 print(output.shape)
#                 print(target.shape)
                loss = self.loss(output,target)
#                 print(loss)

                loss.backward()
                self.optimizer.step()

                if( batch_idx % 20 == 0 ):
#                     print('    loss at batch {}: loss1-{} total-{}'.format(batch_idx, loss1,loss2,loss), flush=True)
                    
                    print('    loss at batch {}: {}'.format(batch_idx, loss), flush=True)


            self.model.eval()
            
            with torch.no_grad(): # very very very very important!!!
                
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
#                     print(loss_val)
                    val_loss.append(loss_val.item())
#                     print(loss_val.item())
#                     val_loss.append(loss_val)
#                     loss_print = torch.mean(torch.stack(val_loss))
                    
                    output = F.softmax(output,dim=0)
                    pred_1 = output.argmax(dim=1, keepdim=True)
                    crr = pred_1.eq( target.view_as(pred_1)  ).sum().item()

                    correct_1 += crr
                  
                    
            corr_1_val = (100. * correct_1) / self.num

            
            print("validation loss : {:.6f}".format( np.mean(val_loss) ))
#             print("validation loss : {:.6f}".format( loss_print ))
#             val_loss = torch.mean(torch.stack(val_loss))    
# #             corr_val = (correct) / len(self.val_datagen)
# #             avg_loss /= len(self.val_datagen)
            self.scheduler.step()
            
#             print("validation loss : %.6f" % avg_loss )
#             print("validation loss2 : %.6f" % val_loss )
            
            print("validation accuracy : %.2f" % corr_1_val) 
        
            PATH_ckpt = self.model_path + "epoch_" + str(epoch+1).zfill(3) + '_ckpt.pth'
            patience = 20
            
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
                    print("____model saved at epoch : {0:d}, accuracy : {1:.2f}".format(epoch+1,val_loss))
#             val_loss = np.mean(val_loss)
#             if best is None :
#                 best = corr_1_val 
#                 counter = 0
#             else : 
#                 if best > corr_1_val:
#                     counter += 1
#                 else :
#                     best = corr_1_val
#                     self.save_model(PATH_ckpt)
#                     counter = 0
#                     print("____model saved at epoch : {0:d}, accuracy : {1:.2f}".format(epoch+1,corr_1_val))
                    
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
        
#     def save_model(self):
#         """Save the final model output."""
#         if self.framework == 'keras':
#             self.model.save(self.config['training']['model_dest_path'])
#         elif self.framework == 'torch':
#             if isinstance(self.model, torch.nn.DataParallel):
#                 torch.save(self.model.module.state_dict(), self.model_path+'final.pth')
#             else:
#                 torch.save(self.model.state_dict(), self.model_path+'final.pth')

"""to print learning schedule"""
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
# def EarlyStopping(loss, val_loss)
#     patience = 10
#     stop = False
#     best = None
#     if best is None :
#         best = val_loss 
#         counter = 0
#     else : 
#         if best < val_loss:
#             counter += 1
    
#     if counter >= patience :
#         stop = True
#     return stop

# class TorchEarlyStopping(object):

#     def __init__(self, patience=10, threshold=0.0, verbose=False):
#         self.patience = patience
#         self.threshold = threshold
#         self.counter = 0
#         self.best = None
#         self.stop = False

#     def __call__(self, metric_score):

#         if self.best is None:
#             self.best = metric_score
#             self.counter = 0
#         else:
#             if self.best - self.threshold < metric_score:
#                 self.counter += 1
#             else:
#                 self.best = metric_score
#                 self.counter = 0

#         if self.counter >= self.patience:
#             self.stop = True
            
# def _run_torch_callbacks(self, loss, val_loss):

#     if isinstance(cb, TorchEarlyStopping):
#         cb(val_loss)
#         if cb.stop:
#             if self.verbose:
#                 print('Early stopping triggered - '
#                       'ending training')
#             return False

#     return True            
            
# def get_callbacks(framework='torch', config):
#     callbacks = []
#     if framework == 'keras':
#         for callback, params in config['training']['callbacks'].items():
#             if callback == 'lr_schedule':
#                 callbacks.append(get_lr_schedule(framework, config))
#             else:
#                 callbacks.append(keras_callbacks[callback](**params))
#     elif framework == 'torch':
#         for callback, params in config['training']['callbacks'].items():
#             if callback == 'lr_schedule':
#                 callbacks.append(get_lr_schedule(framework, config))
#             else:
#                 callbacks.append(torch_callback_dict[callback](**params))

#     return callbacks
            
            
            
            
            
            
            
            
            
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2,alpha=0.25, size_average=True):
# #         super(FocalLoss, self).init()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: 
#             return loss.mean()
#         else: return loss.sum()
        
        
        
class TorchFocalLoss(nn.Module):
    """Implementation of Focal Loss[1]_ modified from Catalyst [2]_ .

    Arguments
    ---------
    gamma : :class:`int` or :class:`float`
        Focusing parameter. See [1]_ .
    alpha : :class:`int` or :class:`float`
        Normalization factor. See [1]_ .

    References
    ----------
    .. [1] https://arxiv.org/pdf/1708.02002.pdf
    .. [2] https://catalyst-team.github.io/catalyst/
    """

    def __init__(self, gamma=2, reduce=True, logits=False):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce
        self.logits = logits

    # TODO refactor
    def forward(self, outputs, targets):
        """Calculate the loss function between `outputs` and `targets`.

        Arguments
        ---------
        outputs : :class:`torch.Tensor`
            The output tensor from a model.
        targets : :class:`torch.Tensor`
            The training target.

        Returns
        -------
        loss : :class:`torch.Variable`
            The loss value.
        """

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets,
                                                          reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(outputs, targets,
                                              reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss        