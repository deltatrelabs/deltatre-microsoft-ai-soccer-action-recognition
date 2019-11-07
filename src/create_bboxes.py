# This is the script to calculate Players Boxes

from config import *

import mPyPl as mp
from mpyplx import *
from pipe import Pipe
from functools import partial

import numpy as np
import cv2
import itertools
from moviepy.editor import *
import pickle

import keras
import tensorflow as tf
import time

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    return tf.Session(config=config)


def get_boxes(model, image, prob_thresh=0.2, class_label=0):
    image = image[:, :, ::-1].copy()
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    boxes = boxes[0][(scores[0] > prob_thresh) & (labels[0] == class_label)]
    return boxes


def calc_image(filename, new_filename, model, get_func, fps=5):
    print("Processing {}".format(filename))
    clp = VideoFileClip(filename)
    frames = list(clp.iter_frames())[::fps]
    boxes = [ get_func(model, frame) for frame in frames] 
    pickle.dump(boxes, open(new_filename, 'wb'))
    clp.reader.close()
    clp.audio.reader.close_proc()


if __name__ == "__main__":
    retina_model = models.load_model(retina_path, backbone_name='resnet50')
    pcalc_retina = partial(calc_image, model=retina_model, get_func=get_boxes)

    stream = (
        mp.get_datastream(data_dir, ext=".full.mp4", classes={'noshot' : 1, 'shot': 2, 'attack' : 0})
        | mp.select_field('filename')
        | cachecompute(".full.mp4",".boxes.pickle", pcalc_retina, lambda x, nx: print("Skipping {}".format(x)))    
        | execute
    )