# Arda Mavi

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return

def get_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', strides=1, input_shape=(64, 64, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', strides=1))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(192, (3, 3)))
    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    return model

if __name__ == '__main__':
    save_model(get_model())
