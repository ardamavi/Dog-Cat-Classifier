# Arda Mavi

import os
from keras.callbacks import ModelCheckpoint

def train_model(model, X, X_test, Y, Y_test):

    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    Checkpoint = ModelCheckpoint('Data/Checkpoints/best_weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    model.fit(X, Y, batch_size=10, epochs=25, validation_data=(X_test, Y_test), shuffle=True, callbacks=[Checkpoint])

    """
    # For better yield. The duration of the training is extended.

    from keras.preprocessing.image import ImageDataGenerator
    generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)
    import numpy
    model.fit_generator(generated_data.flow(X, Y, batch_size=10), steps_per_epoch=X.shape[0], epochs=25, validation_data=(X_test, Y_test), callbacks=[Checkpoint])
    """

    return model

def main():
    from get_model import get_model, save_model
    model = get_model()
    from get_dataset import get_dataset
    X, X_test, Y, Y_test = get_dataset()
    import numpy
    model = train_model(model, X, X_test, Y, Y_test)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
