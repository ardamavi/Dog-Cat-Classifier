# Arda Mavi

from keras.models import Sequential
from keras.models import model_from_json
from os.path import join
import numpy as np
import threading
import cv2
import random

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y

def image_process(model, img):
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    Y = predict(model, X)
    print('It is a ' + Y + ' !')
    return Y



def read_model(model_dir_path):
    # Getting model:
    model_file = open(join(model_dir_path, 'model.json'), 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights(join(model_dir_path, 'weights.h5'))
    return model


def process_and_store_frame(model, lock, name_lock):
    import time
    from get_dataset import img_resize
    ret, frame = get_frame(lock)
    while ret:
        img = img_resize(frame)
        color_conv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Y = image_process(model, color_conv)
        path = join('output', f'{Y}s', f'{str(random.randint(1, 10000))}.jpg')
        cv2.imwrite(path, img)
        time.sleep(10)
        ret, frame = get_frame(lock)


def get_frame(lock):
    with lock:
        if cap.isOpened():
            return cap.read()
    return (False, None)


def process_video_frames(model):
    lock = threading.Lock()
    name_lock = threading.Lock()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    threads = []

    for i in range(int(frame_count / 6)):
        process = threading.Thread(target=process_and_store_frame, args=(model, lock, name_lock))
        threads.append(process)
        print(f'{i} is starting')
        process.start()

    for thread_wait in threads:
        thread_wait.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    import pathlib

    model = read_model('Data/Model')
    input_path = sys.argv[1]
    file_ext = pathlib.Path(input_path).suffix

    if file_ext == '.mp4':
        cap = cv2.VideoCapture(input_path)
        process_video_frames(model)
    else:
        from get_dataset import get_img
        img = get_img(input_path)
        image_process(model, img)


