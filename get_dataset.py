# Arda Mavi

import numpy as np
from os import listdir
from skimage import color, io
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle, to_categorical

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
    for label in labels:
        datas_path = dataset_path+'/'+label
        for data in listdir(datas_path):
            img = get_img(datas_path+'/'+data)
            X.append(img)
            Y.append(label)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42) # Create dateset
    # Shuffle datas:
    X, Y = shuffle(X, Y)
    # Encode Y:
    Y = to_categorical(Y, 2)
    Y_test = to_categorical(Y_test, 2)
    return X, X_test, Y, Y_test
