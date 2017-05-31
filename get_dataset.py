# Arda Mavi

import numpy as np
from os import listdir
from skimage import io
from scipy.misc import imresize

def get_img(data_path):
    # Getting image array from path:
    img_size = 64
    img = io.imread(data_path)
    img = imresize(img, (img_size, img_size, 3))
    return np.array(img)

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    labels = listdir(dataset_path) # Geting labels
    len_datas = 0
    for label in labels:
        len_datas += len(listdir(dataset_path+'/'+label))
    X = np.zeros((len_datas, 64, 64, 3), dtype='float64')
    Y = np.zeros(len_datas)
    count_data = 0
    count_categori = [-1,''] # For encode labels
    for label in labels:
        datas_path = dataset_path+'/'+label
        for data in listdir(datas_path):
            img = get_img(datas_path+'/'+data)
            X[count_data] = img
            # For encode labels:
            if label != count_categori[1]:
                count_categori[0] += 1
                count_categori[1] = label
                print('Categori:\n', count_categori)
            Y[count_data] = count_categori[0]
    # Create dateset:
    test_size = int(len(Y)*0.9)
    X, X_test = X[:test_size], X[test_size:]
    Y, Y_test = Y[:test_size], Y[test_size:]
    return X, X_test, Y, Y_test
