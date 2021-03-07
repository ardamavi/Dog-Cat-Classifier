# Arda Mavi
import numpy as np
import sys
from keras.models import Sequential
from keras.models import model_from_json


class Predictor:
    model = None

    def __init__(self, model_file='Data/Model/model.json'):
        model_file = open('Data/Model/model.json', 'r')
        self.model = model_file.read()
        model_file.close()
        self.model = model_from_json(self.model)

    def predict(self, img):
        X = np.zeros((1, 64, 64, 3), dtype='float64')
        X[0] = img
        Y = self.model.predict(X)
        Y = np.argmax(Y, axis=1)
        Y = 'cat' if Y[0] == 0 else 'dog'
        return Y


def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y


if __name__ == '__main__':

    img_dir = r"Data\Train_Data\cat\cat.0.jpg" #sys.argv[1]

    from get_dataset import get_img
    img = get_img(img_dir)
    print(img)
    print(img.shape)

    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    print('It is a ' + Y + ' !')
