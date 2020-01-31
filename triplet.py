import numpy as np

#return the puller index of the training set
def puller(train_data, train_label, db_data, db_label,num_of_cl = 5):
	N = train_label.shape[0]
	puller_index = np.zeros(N,dtype = int)
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

def pusher(train_data, train_label, db_data, db_label, puller_index, num_of_cl = 5, p = 9.5):
	N = train_label.shape[0]
	pusher_index = np.zeros(N,dtype = int)
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
	    	k=np.random.binomial(1,p,None)
            if k==1:
                pusher_index[j]=np.random.choice(np.delete(idx,puller_index[j]),size=1,replace=True,p=None)
            else:
                pusher_index[j]=np.random.choice(idnx_db,size=1,replace=True,p=None)


    return pusher_index
    
   