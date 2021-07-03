import cv2
import torch
import pandas as pd
import numpy as np
from model import Res2Net
from model_block import Res2Net as RN
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy

import time

now = time.localtime()
# read the video data from mp4 file

# video_num = 2
for video_num in range(1,24):
    cap = cv2.VideoCapture('./data/video/getMedia'+str(video_num)+'.mp4')
    model_path = 'result/epoch_037_ckpt.pth'
    model_path_2 = 'result/final_block_0616.pth'

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # define the writer variables - set the output video file name
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
    out_dir = './result/video/'
    os.makedirs(out_dir,exist_ok=True)
    day = "_"+str(now.tm_mon)+"_"+str(now.tm_mday)+"_"
    _out = cv2.VideoWriter(out_dir+'prediction'+str(video_num)+day+'.mp4', fourcc, fps, (int(width), int(height)))
    df =  pd.read_csv('./data/train.txt')
    df = df.loc[:,['class','place']]
    df = df.drop_duplicates()
    diction  = df.set_index('class', drop = False)
    diction2  = copy.deepcopy(diction)

    diction2['place'][0] = 'blocked'
    diction2['place'][1] = 'no_block'



    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = Res2Net().to(device)
    model2 = RN().to(device)

    model.load_state_dict(torch.load(model_path,map_location=device))
    model2.load_state_dict(torch.load(model_path_2,map_location=device))


    # In[ ]:


    print("start prediction the video work ================ ")
    print("video number : %d" % video_num)
    save_image_set = []
    model.eval()
    model2.eval()
    count =0

    block_count = 0
    with torch.no_grad():
        score_list = []
        place_list = []
        place_cs1=[]
        topk_cs1=[]
        place_cs2=[]
        topk_cs2=[]
        place_cs3=[]
        topk_cs3=[]
        place_cs4=[]
        topk_cs4=[]

        while cap.isOpened(): # check cap is ordinary working
            count+=1
            ret, frame_origin = cap.read()

            if not ret:
                print("Cannot load the Frame data...")
                break
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
            output_2 = model2(data) 

            prob = F.softmax(output_,dim=1)
            prob_2 = F.softmax(output_2,dim=1)

            score_,____ = prob.topk(3,dim=1)
            score_22, pred_indices = torch.topk(output_, 3 )
            _, pred_indices_block = torch.topk(output_2, 1 )

            prediction_name_idx = diction.loc[pred_indices[0][0].cpu().numpy().astype(int)]
            blockornot = diction2.loc[pred_indices_block[0][0].cpu().numpy().astype(int)]

            top_k_score = score_[0][0].cpu().numpy().astype(float)

            prediction_name  = prediction_name_idx[1]
            block  = blockornot[1]

            # case 1 : no block, no cfd
            place_cs1.append(prediction_name) 
            topk_cs1.append(top_k_score)
            orig_place = prediction_name
            orig_topk = top_k_score
            orig_place_2 = prediction_name
            orig_topk_2 = top_k_score
            if count==1 :
                save_place = prediction_name
                save_score = top_k_score
                save_place_2 = prediction_name
                save_score_2 = top_k_score
                save_place_3 = prediction_name
                save_score_3 = top_k_score
                save_place_4 = prediction_name
                save_score_4 = top_k_score

                pred1 = prediction_name 
                pred2 = prediction_name             

            # case 2 : no block, yes cfd
            if top_k_score > 0.98:
                save_place_2 = prediction_name
                save_score_2 = top_k_score
            else : 
                prediction_name = save_place_2
                top_k_score = save_score_2
            place_cs2.append(prediction_name)
            topk_cs2.append(top_k_score)

            # case 3 : yes block, no cfd
            if block != 'blocked':
                save_place_3 = orig_place
                save_score_3 = orig_topk
            else :
                orig_place = save_place_3
                orig_topk = save_score_3

            place_cs3.append(orig_place)
            topk_cs3.append(orig_topk)

            #case 4 : yes block, yes cfd
            if block == 'blocked':
                block_count +=1
                orig_place_2 = save_place_4
                orig_topk_2 = save_score_4
            else :
                if top_k_score > 0.98:
                    save_place_4 = orig_place_2
                    save_score_4 = orig_topk_2
                else : 
                    orig_place_2 = save_place_4
                    orig_topk_2 = save_score_4
            place_cs4.append(orig_place_2)
            topk_cs4.append(orig_topk_2)

            pred1 = prediction_name 
            if pred1 == pred2:
                prediction_name = pred1

            ## input to video section
    #         text = str(prediction_name)
    #         text2 = str(top_k_score)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_RGB2BGR)
    #         cv2.putText(frame_origin, text, (50, 50), font, 2, (0, 0, 255), 4)
    #         cv2.putText(frame_origin, text2, (700, 50), font, 2, (0, 0, 255), 4)
    #         place_list.append(text)
    #         score_list.append(text2)
    #         _out.write(frame_origin)
    csv_dir = './result/result_csv/'
    os.makedirs(csv_dir,exist_ok=True)
    df1_a = pd.DataFrame(place_cs1,columns=['predicted place'])
    df1_b = pd.DataFrame(topk_cs1,columns=['score'])
    df1 = pd.concat([df1_a,df1_b], axis=1)

    df2_a = pd.DataFrame(place_cs2,columns=['predicted place'])
    df2_b = pd.DataFrame(topk_cs2,columns=['score'])
    df2 = pd.concat([df2_a,df2_b], axis=1)

    df3_a = pd.DataFrame(place_cs3,columns=['predicted place'])
    df3_b = pd.DataFrame(topk_cs3,columns=['score'])
    df3 = pd.concat([df3_a,df3_b], axis=1)

    df4_a = pd.DataFrame(place_cs4,columns=['predicted place'])
    df4_b = pd.DataFrame(topk_cs4,columns=['score'])
    df4 = pd.concat([df4_a,df4_b], axis=1)

    file_name = csv_dir+'{}_{}_{}_'.format(now.tm_mon, now.tm_mday,video_num)
    df1.to_csv(file_name+'case1.csv')
    df2.to_csv(file_name+'case2.csv')
    df3.to_csv(file_name+'case3.csv')
    df4.to_csv(file_name+'case4.csv')
    try : 
        print('block ratio : {0:d}/{1:d} = {2:.4f}'.format(block_count,count,100*block_count/count))
    except :
        print('no block')
        pass
    

    # df_3.to_csv('./place_result_final_block_2_ver2.csv')


    cap.release()
    _out.release()
    cv2.destroyAllWindows()







