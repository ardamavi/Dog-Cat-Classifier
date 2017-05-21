# Arda Mavi

import os
import tflearn
from tflearn.metrics import Accuracy
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.core import input_data, dropout, fully_connected

def save_model(model, name='model'):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    save_as = 'Data/Model/'+name+'.tflearn'
    model.save(save_as)
    print('Model saved as \''+save_as+'\'')
    return

def get_model():
    # Normalisation:
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # For synthetic data:
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Network Architecture:
    network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

    # 1: Convolution layer: 32 filters each 3x3x3:
    conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

    # 2: Max pooling:
    network = max_pool_2d(conv_1, 2)

    # 3: Convolution layer: 64 filters:
    conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

    # 4: Convolution layer: 64 filters:
    conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

    # 5: Max pooling:
    network = max_pool_2d(conv_3, 2)

    # 6: Fully-connected: 512 node:
    network = fully_connected(network, 512, activation='relu')

    # 7: Dropout:
    network = dropout(network, 0.5)

    # 8: Fully-connected: 2 outputs:
    network = fully_connected(network, 2, activation='softmax')

    # Train configure:
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0005, metric=acc)

    # Create model:
    if not os.path.exists('Data/Logs/'):
        os.makedirs('Data/Logs/')
    model = tflearn.DNN(network, checkpoint_path='checkpoint.tflearn', max_checkpoints = 3, tensorboard_verbose = 3, tensorboard_dir='Data/Logs/')

    return model

if __name__ == '__main__':
    save_model(get_model())
