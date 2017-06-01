# Arda Mavi

def train_model(model, X, X_test, Y, Y_test):
    model.fit(X, Y, batch_size=32, epochs=25, validation_data=(X_test, Y_test), shuffle=True)
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
