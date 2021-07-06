import cv2
import torch
import pandas as pd
import numpy as np
from model import Res2Net
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy

import time
from config import parse
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
config_path = './param_config.yml'
config = parse(config_path)
now = time.localtime()
'''
Read the video data from mp4 file.
Setup the video directory, output directory, and file name(***.mp4)

'''

video_dir = './video/'
file_name = config['data']['video_file_name']
out_dir = config['result_path']

cap = cv2.VideoCapture(video_dir+file_name+'.mp4')
model_path = config['result_path'] + '/'+config['weight_file']
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

# define the writer variables - set the output video file name
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # define the codec
os.makedirs(out_dir,exist_ok=True)
day = "_"+str(now.tm_mon)+"_"+str(now.tm_mday)+"_"
_out = cv2.VideoWriter(out_dir+'prediction'+file_name+day+'.mp4', fourcc, fps, (int(width), int(height)))
df =  pd.read_csv('./data/train.txt')
df = df.loc[:,['class','place']]
df = df.drop_duplicates()
diction  = df.set_index('class', drop = False)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = Res2Net(config).to(device)
model.load_state_dict(torch.load(model_path,map_location=device))


print("start prediction the video work ================ ")
print("file_name : %s" % file_name)
save_image_set = []
model.eval()

count =0
with torch.no_grad():
    score_list = []
    place_list = []

    while cap.isOpened(): # check cap is ordinary working
        count+=1
        ret, frame_origin = cap.read()

        if not ret:
            print("Cannot load the Frame data...")
            print("Prediction completed")
            break
            # if you just want to check the code work or not, you can stop at 100th frame.
#         if count==100:
#             break
        frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
        if count % 100 ==0:
            print("In progress.... count = ",count)

        img_arr= cv2.resize(frame_origin,(256,256))
        img_arr = img_arr /255

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
        

        score_,____ = prob.topk(3,dim=1)
        score_22, pred_indices = torch.topk(output_, 3 )
        

        prediction_name_idx = diction.loc[pred_indices[0][0].cpu().numpy().astype(int)]

        top_k_score = score_[0][0].cpu().numpy().astype(float)

        prediction_name  = prediction_name_idx[1]


        ## input to video section
        text = '{0}'.format(prediction_name)
        text2 = '{0:.4f}'.format(top_k_score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_origin, text, (50, 50), font, 2, (0, 0, 255), 4)
        cv2.putText(frame_origin, text2, (700, 50), font, 2, (0, 0, 255), 4)
        place_list.append(text)
        score_list.append(text2)
        _out.write(frame_origin)
csv_dir = './result/result_csv/'
os.makedirs(csv_dir,exist_ok=True)
df1_a = pd.DataFrame(place_list,columns=['predicted place'])
df1_b = pd.DataFrame(score_list,columns=['score'])
df1 = pd.concat([df1_a,df1_b], axis=1)


file_name = csv_dir+'{}_{}_{}_'.format(now.tm_mon, now.tm_mday,file_name)
df1.to_csv(file_name+'.csv')

cap.release()
_out.release()
cv2.destroyAllWindows()







