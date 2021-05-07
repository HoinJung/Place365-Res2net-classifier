#!/usr/bin/env python
# coding: utf-8

# In[2]:


# read the video data from mp4 file
import cv2
import torch
import pandas as pd
import numpy as np
from model import Res2Net
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# # os.environ['CUDA_VISIBLE_DEVICES']='1'

# In[3]:


# read the video data from mp4 file
cap = cv2.VideoCapture('./video/getMedia14.mp4')
model_path = 'result/final.pth'
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # or cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # or cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # or cap.get(5)

# define the writer variables - set the output video file name
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
_out = cv2.VideoWriter('./video/prediction14_ver1.mp4', fourcc, fps, (int(width), int(height)))
df =  pd.read_csv('./data/train.txt')
df = df.loc[:,['class','place']]
df = df.drop_duplicates()
diction  = df.set_index('class', drop = False)

# In[4]:



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
# device = torch.device('cuda')
model = Res2Net().to(device)

model.load_state_dict(torch.load(model_path,map_location=device))


# In[ ]:


print("start prediction the video work ================ ")
save_image_set = []
model.eval()
count =0


with torch.no_grad(): 
    while cap.isOpened(): # check cap is ordinary working
        count+=1
        ret, frame_origin = cap.read()

        if not ret:
            print("Cannot load the Frame data...")
            break

        frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
        if count % 100 ==0:
            print("In progress.... count = ",count)

        img_arr= cv2.resize(frame_origin,(256,256))
        img_arr = img_arr /255
#         img_arr= cv2.resize(img_arr,(256,256))
#         print(img_arr)
        shape_val = img_arr.shape

        if len(shape_val) == 3 :
            x_input = torch.FloatTensor( np.transpose(img_arr, (2, 0, 1)) )
        else :
            x_input = torch.FloatTensor( [img_arr, img_arr, img_arr] )
        frame = torch.FloatTensor(x_input)

        frame = frame.unsqueeze(0)      
        data = frame.to(device)

        output_ = model(data) 

        prob = F.softmax(output_,dim=1)

        score_,____ = prob.topk(5,dim=1)
        score_22, pred_indices = torch.topk(output_, 5 )

        prediction_name_idx = diction.loc[pred_indices[0][0].cpu().numpy().astype(int)]
        
        top_k_score = score_[0][0].cpu().numpy().astype(float)
        prediction_name  =prediction_name_idx[1]
        # algorithm 1 ## 저장도안됨
#         if top_k_score < 0.7:
#             continue
        # algorithm 2
        if count==1 :
            save_place = prediction_name
            save_score = top_k_score
        if top_k_score > 0.9:
            save_place = prediction_name
            save_score = top_k_score
        else : 
            prediction_name = save_place
            top_k_score = save_score
            
        if prediction_name == 'bedroom' :
            prediction_name = 'home'
        elif prediction_name == 'living_room' :
            prediction_name = 'home'
#         if count ==200:
#             break
  
        # none algorithm
        text = str(prediction_name)
        text2 = str(top_k_score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_origin, text, (50, 50), font, 2, (255, 255, 255), 4)
        cv2.putText(frame_origin, text2, (700, 50), font, 2, (255, 255, 255), 4)
        
        _out.write(frame_origin)

cap.release()
_out.release()
cv2.destroyAllWindows()







