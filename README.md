# Res2net_classifier
## Introduction 
- This model based on Res2Net("Res2net: A new multi-scale backbone architecture.").   
- The purpose of this model is classifying 'place' in image or video.   

## Dataset
### Released Version
- I adopted **Place365** dataset for the train and test.  
- You can download the whole data right here. Link: [PLACE365](http://places2.csail.mit.edu/download.html)    
I prefer 'small image dataset with easy directory structure'.
### Custom Dataset
You can make your own dataset.
- Input training data to
```'./data/train/<label_name>/imageXXX.jpg'```.   
- And input testing data to 
```'./data/val/<label_name>/imageXXX.jpg'```                
In this case, we use validation images of given dataset as *testing set*.   

- And you don't need additional validation dataset because we use *validation split*.   
You can adjust ```val_holdout_frac``` in ```param_config.yml```.   

### Create data addressing file
    python data_prep.py
to make csv label file to train and test.

## Training
- Setup batchsize, number of classes, epoch, learning rate, optimizer in ```param_config.yml```.   
- We adopt cross entropy loss. If you want to change loss function, modify it manually in ```train.py```.    
To start train, 

      python run_train.py
      
The ```run_train.py``` execute ```train.py```.

## Inference
### Test for the images in dataset.
To start test

     python test.py
     
You can show the result top-5 prediction and accuracy.
And the result will be saved as ```csv``` file in ```./result_csv/```.
    
### Test for the video.    
Make directory ```./video/``` and input a video you want to predict.
Run

     video_prediction.py

The predicted data would be saved at ```./result/video/``` and ```./result_csv/```.

* * *

## Reference

    1. Gao, Shanghua, et al. "Res2net: A new multi-scale backbone architecture." IEEE transactions on pattern analysis and machine intelligence (2019).
    
    2. Places: A 10 million Image Database for Scene Recognition B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017

