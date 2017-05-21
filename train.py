# Arda Mavi

import tflearn
from get_model import save_model

def train_model(model, X, X_test, Y, Y_test):
    model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500, n_epoch=100, run_id='model', show_metric=True)
    return model

def main():
    from get_model import get_model
    model = get_model()
    from get_dataset import get_dataset
    X, X_test, Y, Y_test = get_dataset()
    model = train_model(model, X, X_test, Y, Y_test)
    model.save_model(model)
    return model

if __name__ == '__main__':
    main()
