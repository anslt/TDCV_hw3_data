import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def batch_generator(p,num_per_cl,match_matrix,train_data,train_label,db_data,db_label,num_of_cl=5):
    batch_mm=np.zeros((num_per_cl*num_of_cl,3),dtype=np.int)
    
    for i in range(num_of_cl):
        st=np.array(np.where(match_matrix[:,1337]==i))
        st=st.reshape(st.size,)
        batch_mm[i*num_per_cl:(i+1)*num_per_cl,0]=np.random.choice(st,size=num_per_cl,replace=False,p=None) ## row number of class i in mm
    
    batch_mm[:,1]=match_matrix[batch_mm[:,0],1]
    
    for j in range(0,batch_mm.shape[0]):
        k=np.random.binomial(1,p,None)
        if k==1:
            batch_mm[j,2]=np.random.choice(match_matrix[batch_mm[j,0],2:268],size=1,replace=True,p=None)
        else:
            batch_mm[j,2]=np.random.choice(match_matrix[batch_mm[j,0],268:1336],size=1,replace=True,p=None)
            
            
            

    batch_X=np.zeros((num_per_cl*15,64,64,3),dtype=np.float32)
    batch_y=np.zeros((num_per_cl*15,5),dtype=np.float32)

    t=0
    for i in range(0,batch_mm.shape[0]):
        batch_X[t,:,:,:]=train_data[batch_mm[i,0],:,:,:]
        batch_X[t+1,:,:,:]=db_data[batch_mm[i,1],:,:,:]
        batch_X[t+2,:,:,:]=db_data[batch_mm[i,2],:,:,:]

        batch_y[t,:]=train_label[batch_mm[i,0],:]
        batch_y[t+1,:]=db_label[batch_mm[i,1],:]
        batch_y[t+2,:]=db_label[batch_mm[i,2],:]
        t=t+3
    return batch_X,batch_y
