# Arda Mavi

from keras.models import Sequential
from keras.models import model_from_json

def predict(model, X):
    return model.predict(X)

if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    from get_dataset import get_img
    X = get_img(img_dir)
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model_file = model_file.read()
    model_file.close()
    model = model_from_json(model_file)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    print(predict(model, X))
