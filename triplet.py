import numpy as np

#return the puller index of the training set
def puller_construct(train_data, train_label, db_data, db_label,num_of_cl = 5):
    N = train_label.shape[0]
    puller_index = np.zeros(N, dtype = int)
    for i in range(num_of_cl):
    ## find the row number of samples with label i in train and database data
        idx_tr = np.array(np.where(train_label[:,0]==i))
        idx_tr = idx_tr.reshape(idx_tr.shape[1],)
        idx_db = np.array(np.where(db_label[:,0]==i))
        idx_db = idx_db.reshape(idx_db.shape[1],)

        ## see the distance of any match of database data and training data
        dot_matrix=np.dot(train_label[idx_tr,1:5],db_label[idx_db,1:5].T)
        xita_matrix= 2 * np.arccos(np.minimum(abs(dot_matrix),1)) ## some error in 562,163 dot product leads to 1.02

        puller_index[idx_tr] = idx_db[np.argmin(xita_matrix,axis=1)]
    return puller_index

def pusher_construct(train_data, train_label, db_data, db_label, puller_index, num_of_cl = 5):
    N = train_label.shape[0]
    pusher_index = np.zeros([N,2],dtype = int)
    for i in range(num_of_cl):
    ## find the row number of samples with label i in train and database data
        idx_tr = np.array(np.where(train_label[:,0]==i))
        idx_tr = idx_tr.reshape(idx_tr.shape[1],)
        idx_db = np.array(np.where(db_label[:,0]==i))
        idx_db = idx_db.reshape(idx_db.shape[1],)
        idnx_db = np.array(np.where(db_label[:,0]!=i))
        idnx_db = idnx_db.reshape(idnx_db.shape[1],)

    #find pusher randomly 
        for j in idx_tr:
            pusher_index[j,0]=np.random.choice(idnx_db,size=1,replace=True,p=None)
            pusher_index[j,1]=np.random.choice(np.delete(idx_db, np.where(idx_db==puller_index[j])),size=1,replace=True,p=None)
                   
    return pusher_index

def triplet_construct(train_data, train_label, db_data, db_label, p = 0.5, old_triplet = None):
    if old_triplet == None:
        puller_index = puller_construct(train_data, train_label, db_data, db_label)
    else:
        puller_index = old_triplet[:,0]

    pusher_index = pusher_construct(train_data, train_label, db_data, db_label, puller_index)
    pusher_choice = np.random.binomial(1, p, train_label.shape[0])
    pusher_index = pusher_index[np.arange(train_label.shape[0]),pusher_choice]
    ans = np.zeros((puller_index.shape[0],3), dtype = int)
    ans[:,0] = np.arrange(train_label.shape[0])
    ans[:,1] = puller_index
    ans[:,2] = pusher_index
    return ans

def batch_generator_from_triplet(batch_size, triplet, train_data, train_label, db_data, db_label):

    batch_X=np.zeros(batch_size * 3,64,64,3),dtype=np.float32)
    batch_y=np.zeros(batch_size * 3,5),dtype=np.float32)

    batch_X[np.arange(batch_size) * 3,:,:,:]=train_data[triplet[:,0],:,:,:]
    batch_X[np.arange(batch_size) * 3 + 1,:,:,:]=train_data[triplet[:,1],:,:,:]
    batch_X[np.arange(batch_size) * 3 + 2,:,:,:]=train_data[triplet[:,2],:,:,:]

    batch_y[np.arange(batch_size) * 3 ,:]=train_data[triplet[:,0],:]
    batch_y[np.arange(batch_size) * 3 + 1,:]=train_data[triplet[:,1],:]
    batch_y[np.arange(batch_size) * 3 + 2,:]=train_data[triplet[:,2],:]


    return batch_X,batch_y
    
   