# Arda Mavi

import tflearn

def predict(model, X):
    return model.predict(X)

if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    from get_dataset import get_img
    X = get_img(img_dir)
    try:
        from get_model import get_model
        model = get_model()
        model = model.load("Data/Model/model.tflearn")
    except:
        import train
        model = train.main()
    print(predict(model, X))
