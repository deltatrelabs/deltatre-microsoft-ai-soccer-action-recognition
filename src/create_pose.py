# This is the script to calculate Dense Poses
# Should be run after resize_video

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

import tensorflow as tf

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def get_poses(model, image, shape=(432, 368)):
    w, h = shape
    if (image.shape[1] < pose_min_width) or (image.shape[0] < pose_min_height):
        return []
    image = cv2.resize(image, shape)
    humans = model.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    return humans
    
    
def calc_sub(filename, new_filename, model, get_func):
    print("Processing {}".format(filename))
    clp = VideoFileClip(filename)
    frames = list(clp.iter_frames())
    boxes = pickle.load(open(filename.replace('.resized.mp4','.boxes.pickle'), 'rb'))
    poses = []
    for f, bs in zip(frames, boxes):
        fposes = []
        for box in bs:
            x1,y1,x2,y2 = box.astype(int)
            pad = abs(x2 - x1) * 0.2
            sub = f[max(y1, y1-30):max(y2, y2+30),min(x1, x1-30):max(x2,x2+30)]
            fposes.append(get_func(model, sub))
        poses.append(fposes)
    pickle.dump(poses, open(new_filename, 'wb'))                        
    clp.reader.close()
    clp.audio.reader.close_proc()


# Dense Pose Calculation

pose_model = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368))
pcalc_pose = partial(calc_sub, model=pose_model, get_func=get_poses)

stream = (
    mp.get_datastream(data_dir, ext='.resized.mp4')
    | mp.select_field('filename')
    | cachecompute('.resized.mp4','.poses.pickle', pcalc_pose, lambda x, nx: print("Skipping {}".format(x)))    
    | execute
)