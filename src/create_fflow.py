# This script creates focused optical flow from resized videos

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

from focused import *


def calc_fflow(filename, new_filename, fflow, fps=5):
    print("Processing {}".format(filename))
    clp = VideoFileClip(filename)
    frames = list(clp.iter_frames())[::fps]
    boxes = pickle.load(open(filename.replace('full.mp4', 'boxes.pickle'), "rb"))
    flow = fflow.run(frames, boxes)
    pickle.dump(flow, open(new_filename, 'wb'))
    clp.reader.close()
    clp.audio.reader.close_proc()

        
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

fflow = FocusedFlow(feature_params, lk_params)
pcalc_fflow = partial(calc_fflow, fflow=fflow, fps=5)

stream = (
    mp.get_datastream(data_dir, ext=".full.mp4", classes={'noshot' : 1, 'shot': 2, 'attack' : 0})
    | mp.select_field('filename')
    | cachecompute(".full.mp4",".fflow.pickle", pcalc_fflow, lambda x, nx: print("Skipping {}".format(x)))    
    | execute
)