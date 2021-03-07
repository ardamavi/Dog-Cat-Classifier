import os
import numpy as np
import cv2
import traceback
import atexit
from concurrent.futures import ThreadPoolExecutor
from get_dataset import imresize
from predict import Predictor


class VideoProcessor:
    cap = None
    pre_process_pipeline = []
    predictor = None
    post_process_actions = []

    def __init__(self, process_pipeline=[], predictor=None, post_process_actions=[]):
        self.pre_process_pipeline = process_pipeline
        self.predictor = predictor
        self.post_process_actions = post_process_actions

    def process(self, video_path='vtest.avi'):
        self.cap = cv2.VideoCapture(video_path)
        i = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if frame is None:
                "eof"
                break

            for func in self.pre_process_pipeline:
                frame = func(frame)

            if self.predictor:
                prediction = self.predictor(frame)

                for func in self.post_process_actions:
                    func(prediction, frame)

            try:
                cv2.imshow('frame', frame)
            except cv2.error:
                traceback.print_exc()
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


class FileWriter:
    f = None

    def __init__(self, path="output/task2.txt"):
        self.f = open(path, mode='w', encoding='utf-8')
        atexit.register(self.cleanup)

    def write(self, string):
        self.f.write(string + '\n')

    def cleanup(self):
        self.f.flush()
        self.f.close()


class AsyncCatDogSaver:
    thread_pool = None
    cat_dir = None
    dog_dir = None
    index = 0

    def __init__(self, cat_dir="output/cats", dog_dir="output/dogs"):
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        self.cat_dir = cat_dir
        self.dog_dir = dog_dir

    def save_frame(self, filename, img):
        print(filename)
        cv2.imwrite(filename, img)

    def save_cat_dog_async(self, prediction, img):
        out_dir = self.dog_dir if prediction == 'dog' else self.cat_dir
        self.index = self.index + 1
        self.thread_pool.submit(self.save_frame, "{out_dir}/{index}.jpg".format(out_dir=out_dir, index=self.index), img)

    def shutdown(self, wait=True):
        self.thread_pool.shutdown(wait)


if __name__ == '__main__':

    pipeline = [
                lambda img: imresize(img, (64, 64, 3)),
                lambda img: cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
               ]
    predictor = Predictor()
    fw = FileWriter()
    async_saver = AsyncCatDogSaver()
    post_process_actions = [lambda prediction, img: print(prediction),
                            lambda prediction, img: fw.write(prediction),
                            lambda prediction, img: async_saver.save_cat_dog_async(prediction, img)]

    video_processor = VideoProcessor(process_pipeline=pipeline,
                                     predictor=predictor.predict,
                                     post_process_actions=post_process_actions)

    video_processor.process(video_path=r"Data/Videos/cats_and_dogs.mp4")
    async_saver.shutdown()

# cap = cv2.VideoCapture('vtest.avi')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()