# Arda Mavi

import numpy as np
from os import listdir
from skimage import color, io
from scipy.misc import imresize
from tflearn.data_utils import shuffle
from sklearn.cross_validation import train_test_split

def get_img(data_path):
    # Getting image array from path:
    img_size = 64
    img = io.imread(data_path)
    img = imresize(img, (img_size, img_size, 3))
    return np.array(img)

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    labels = listdir(dataset_path) # Geting labels
    X, Y = [], []
    count = [0,labels[0]] # For encode labels
    for label in labels:
        datas_path = dataset_path+'/'+label
        for data in listdir(datas_path):
            img = get_img(datas_path+'/'+data)
            X.append(img)
            # For encode labels:
            if label != count[1]:
                count[0] += 1
                count[1] = label
            Y.append(count[0])
    # Shuffle datas:
    X, Y = shuffle(X, Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42) # Create dateset
    return X, X_test, Y, Y_test
