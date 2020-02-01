import matplotlib.pyplot as plt
import numpy as np

def plot_hist(test_predict, test_label, db_predict, db_label, iter):
    arg10 = 0
    arg20 = 0
    arg40 = 0
    arg180 = 0
    for i in range(test_predict.shape[0]):
        dist = test_predict[i] - db_predict
        predict_label = db_label[np.argmin(np.sum(dist ** 2,axis=1))]
        real_label = test_label[i]
        if(predict_label[0] == real_label[0]): 
            arg180 = arg180 + 1
            arg_dist = (2 * np.arccos(np.minimum(abs(np.dot(predict_label[1:5],real_label[1:5])),1))) / np.pi * 180
            if arg_dist < 40:
                arg40 = arg40 + 1
                if arg_dist < 20:
                    arg20 = arg20 + 1
                    if arg_dist < 10:
                        arg10 = arg10 + 1
    arg_list = np.array([arg10, arg20, arg40, arg180]) / test_predict.shape[0]
    print("<10:"+str(arg10) + ", <20:" + str(arg20) + ", <40:" + str(arg40) + ", <180:" + str(arg180))
    plt.bar(["<10","<20","<40","<180"], arg_list)
    plt.title("iteration:" + str(iter))
    plt.ylim(0.0, 1.0)
    plt.show()

def hist_data(test_predict, db_predict, db_label):

    predict_label = np.zeros((test_predict.shape[0],5))
    for i in range(test_predict.shape[0]):
        dist = test_predict[i] - db_predict
        predict_label[i] = db_label[np.argmin(np.max(dist ** 2,axis=1))]

    return predict_label

