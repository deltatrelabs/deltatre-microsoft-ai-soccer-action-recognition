# This script creates audio features from videos
# Should be run after resize_video

from config import *

import mPyPl as mp
from mpyplx import *
from pipe import Pipe
from moviepy.editor import *
import numpy as np
import itertools
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
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

def calcaudio(x,nfn):
    print("Creating audio descriptors for {}".format(x['filename']))
    vid = x['video']
    au = audioBasicIO.stereo2mono(vid.audio.to_soundarray())
    f = audioFeatureExtraction.stFeatureExtraction(au, audio_rate, 0.050 * audio_rate, 0.025 * audio_rate)[0]
    np.save(nfn, f)


if __name__ == "__main__":
    (mp.get_datastream(data_dir,ext=".resized.mp4")
     | load_moviepy_video()
     | cachecomputex(".resized.mp4",".audiofeatures.npy",calcaudio,lambda x,nx: print("Skipping {}".format(x)))
     | close_moviepy_video()
     | execute
    )
