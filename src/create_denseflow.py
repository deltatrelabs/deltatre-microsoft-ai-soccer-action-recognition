# This script creates dense optical flow from resized videos
# Should be run after resize_video

from config import *

import mPyPl as mp
from mpyplx import *
from pipe import Pipe
from moviepy.editor import *
import numpy as np
import itertools

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_dense_flow(frms):
    return np.array([mp.dense_optical_flow(u,v) for u,v in pairwise(frms)])

def calcflow(x,nfn):
    print("Creating dense flow descriptor for {}".format(x['filename']))
    clp = x['video']
    df = get_dense_flow(clp.iter_frames())
    np.save(nfn, df)


if __name__ == "__main__":
    (mp.get_datastream(data_dir,ext=".resized.mp4")
     | load_moviepy_video()
     | cachecomputex(".resized.mp4",".optflow.npy",calcflow,lambda x,nx: print("Skipping {}".format(x['filename'])))
     | close_moviepy_video()
     | execute
    )
