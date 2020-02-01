import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def read_data(path,num_of_pic,pic_name,num_of_cl=5,ob_cl=["/ape","/benchvise","/cam","/cat","/duck"]):
    data=np.zeros((num_of_pic*num_of_cl,64,64,3),dtype=np.float32)
    data_grey=np.zeros((num_of_pic*num_of_cl,64,64),dtype=np.float32)
    t=0
    for j in range(num_of_cl):
        for i in range(num_of_pic):
            imp=path+ob_cl[j]+pic_name+str(i)+".png"
            im=Image.open(imp)
            im_grey=im.convert('L')
            data[t,:,:,:]=np.array(im)
            data_grey[t,:,:]=np.array(im_grey)
            t=t+1
    return data, data_grey


def norm_data(data):
    data_n=np.zeros(data.shape,dtype=np.float32)
    for ch in range(3):
        for i in range(64):
            for j in range(64):
                mean=np.mean(data[:,i,j,ch])
                std=np.std(data[:,i,j,ch])
                data_n[:,i,j,ch]=(data[:,i,j,ch]-mean)/(std+1e-10)
    return data_n

# def norm_data(data):
#     data_n=np.zeros(data.shape,dtype=np.float32)
#     for ch in range(3):
#         for i in range(data.shape[0]):
#             mean=np.mean(data[i,:,:,ch])
#             std=np.std(data[i,:,:,ch])
#             data_n[i,:,:,ch]=(data[i,:,:,ch]-mean)/(std+1e-10)              
#     return data_n    


def read_label(path,num_of_la,num_of_cl=5):
    label=np.zeros((num_of_cl*num_of_la,5),dtype=np.float32)
    ob_cl=["/ape","/benchvise","/cam","/cat","/duck"]
    for i in range(num_of_cl):
        start=num_of_la*i
        end=num_of_la*(i+1)
        label_path=path+ob_cl[i]+"/poses.txt"
        label[start:end,1:5]=np.loadtxt(label_path)
        label[start:end,0]=i
    return label




















