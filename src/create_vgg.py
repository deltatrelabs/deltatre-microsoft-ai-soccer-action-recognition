# This script creates vgg embeddings from resized videos
# Should be run after resize_video

from config import *

import mPyPl as mp
from mpyplx import *
from pipe import Pipe
import numpy as np
import itertools

import keras

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

vgg = keras.applications.vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(video_size[1],video_size[0],3))

def get_vgg(video):
    res = vgg.predict(keras.applications.vgg16.preprocess_input(video))
    return res

def calcvgg(x,nfn):
    print("Creating VGG descriptors for {}".format(x['filename']))
    clp = x['video']
    df = get_vgg(np.array(list(clp.iter_frames())))
    np.save(nfn, df)


if __name__ == "__main__":
    (mp.get_datastream(data_dir,ext=".resized.mp4")
     | load_moviepy_video()
     | cachecomputex(".resized.mp4",".vgg.npy",calcvgg,lambda x,nx: print("Skipping {}".format(x)))
     | close_moviepy_video()
     | execute
    )
