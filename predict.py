# Arda Mavi

from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import cv2
from get_dataset import get_img
import sys
import os
import time
from multiprocessing import Pool, Value
from functools import partial
from skimage.metrics import structural_similarity as compare_ssim

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y


def extract_video_frames(video_path, save_location):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    if len(os.listdir(save_location)) == 200: #cached
        return
    while success:
        cv2.imwrite(save_location + "/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def load_model(model_path, weights_path):
    model_file = open(model_path, 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights(weights_path)
    return model

def save_image(path,img):
    file_name = path + str(time.time())+ ".jpg"
    cv2.imwrite(file_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(file_name + " saved")
    time.sleep(10)


def save_images(name, img_list):
    save_dir = 'output/' + name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = save_dir + "/" + name

    #multiprocessing saving
    with Pool(5) as p:
        p.map(partial(save_image,save_name),img_list)



if __name__ == '__main__':
    imgs = []
    cats = []
    dogs = []
    save_files= True
    model = load_model('Data/Model/model.json', "Data/Model/weights.h5")
    # if input is video
    if '.mp4' in sys.argv[1]:
        #save frames locally - cache
        extract_video_frames(sys.argv[1], 'Data/Frames')
        frames = os.listdir('Data/Frames')
        for frame in frames:
            image = get_img('Data/Frames/' + frame)
            imgs.append(image)
    else:  # else it's image
        img_dir = sys.argv[1]
        image = get_img(img_dir)
        imgs.append(image)
    for img in imgs:
        X = np.zeros((1, 64, 64, 3), dtype='float64')
        X[0] = img
        # Getting model:
        Y = predict(model, X)
        if Y == 'cat':
            cats.append(img)
        else:
            dogs.append(img)
        print('It is a ' + Y + ' !')
    if save_files:
        save_images('cats', cats)
        save_images('dogs', dogs)

#my tries to convert img colors
def try_image_convert():
    import imutils
    imageA = cv2.imread('Data/Train_Data/cat/cat.0.jpg')
    imageB = cv2.imread('Data/Frames/frame0.jpg')
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)