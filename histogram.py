import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import math 

def histogram(model,test_data,test_label,db_data,db_label):
    result=np.zeros((test_data.shape[0],3),dtype=np.float32)
    de_db=model.predict_on_batch(db_data)
    for i in range(0,test_data.shape[0]):
        pre_de=model.predict_on_batch(test_data[i:i+1,:,:,:])
        idx=np.argmin(np.diag(np.matmul(de_db-pre_de,np.transpose(de_db-pre_de))))
        pre_label=db_label[idx]
        gt_label=test_label[i]
        if pre_label[0]==gt_label[0]:
            dot=np.dot(pre_label[1:5],gt_label[1:5].T)
            angle=math.degrees(2 * np.arccos(abs(np.minimum(np.maximum(dot,-1),1))))
            result[i,0]=1
            result[i,1]=angle
            result[i,2]=idx

    his=np.array([0,0,0,0])
    his[0]=np.size(np.where((result[:,1]<10)&(result[:,0]==1)))
    his[1]=np.size(np.where((result[:,1]<20)&(result[:,0]==1)))
    his[2]=np.size(np.where((result[:,1]<40)&(result[:,0]==1)))
    his[3]=np.size(np.where(result[:,0]==1))
    return his
                   
