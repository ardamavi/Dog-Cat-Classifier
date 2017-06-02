# Arda Mavi

def train_model(model, X, X_test, Y, Y_test):
    """
    # For better yield. The duration of the training is extended.
    from keras.preprocessing.image import ImageDataGenerator
    generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)
    model.fit_generator(generated_data.flow(X, Y, batch_size=30), steps_per_epoch=30, epochs=25, validation_data=(X_test, Y_test))
    """
    model.fit(X, Y, batch_size=10, epochs=25, validation_data=(X_test, Y_test), shuffle=True)

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
