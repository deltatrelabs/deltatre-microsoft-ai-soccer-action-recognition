# This script will do all required pre-processing and attribute creation
# For parallel processing, it expects two parameters: number of current node in cluster (from 0 to n-1), and number of nodes n
# If no parameters, it will run on single node

from config import *
import os
import mPyPl as mp
from mpyplx import *
from pipe import Pipe
from moviepy.editor import *
import sys
import functools

import resize_video
import create_denseflow
import create_vgg
import create_audio_features
import create_scene_change

def skip(x,nx,s):
    print("Skipping {} {}".format(s,nx))

def get_scene_changes(resized_file_names, base_dir_pkl = "../etc", create_if_not_found = True):
    out_file = os.path.join(base_dir_pkl, scene_detection_file)
    if os.path.exists(out_file):
        import pickle as pkl
        with open(out_file, "rb") as scene_changes_file:
            scene_changes = pkl.load(scene_changes_file)
    elif create_if_not_found:
        scene_changes = create_scene_change.detect_and_write(resized_file_names, filename = out_file)
    else:
        raise ValueError("Data file '%s' not found" % scene_detection_file)

    return scene_changes

if __name__ == "__main__":

    if (len(sys.argv)>1):
        k = int(sys.argv[1])
        n = int(sys.argv[2])
        config.base_dir = config.base_dir_batch
        config.data_dir = config.data_dir_batch
    else:
        k,n = 0,1

  
    (mp.get_datastream(data_dir,ext=".full.mp4")
     | batch(k,n)
     | mp.fapply('video',resize_video.load_resize)
     | execute
    )

    resized_file_names = (mp.get_datastream(data_dir, ext=".resized.mp4")
     | mp.select_field("filename")
     | mp.as_list
    )

    # use only the first threshold
    scene_changes = get_scene_changes(resized_file_names, data_dir)[40]

    (mp.get_datastream(data_dir, ext=".resized.mp4")
     | mp.filter("filename", lambda f: os.path.abspath(f) not in scene_changes)
     | cachecomputex(".resized.mp4", ".optflow.npy", create_denseflow.calcflow, functools.partial(skip,s="creating dense flow"))
     | cachecomputex(".resized.mp4", ".vgg.npy", create_vgg.calcvgg, functools.partial(skip,s="creating VGG"))
     | cachecomputex(".resized.mp4", ".audiofeatures.npy", create_audio_features.calcaudio, functools.partial(skip,s="creating audio features"))
     | close_moviepy_video()
     | execute
    )
