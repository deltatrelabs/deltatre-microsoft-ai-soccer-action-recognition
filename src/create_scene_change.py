# This script detects all scene changes from resized videos
# Should be run after resize_video

from config import *
import os

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

import mPyPl as mp
from mpyplx import *

import itertools
import functools
import os
import numpy as np
import cv2
import json 
from collections import defaultdict

from multiprocessing import Pool
import pickle as pkl


def show_scenes(scene_lib):
    for file, frames in scene_lib.items():
        cap = cv2.VideoCapture(file)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(file)
        for frame_no in frames:
            for n in range(-1, 1):
                if frame_no + n < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + n)
                    ret, frame = cap.read()
                    while(cap.isOpened()):
                        cv2.imshow("%d" % n, frame)
                        if cv2.waitKey(20) & 0xFF == ord('q'):
                            break
            cv2.destroyAllWindows()


        cap.release()


def process(files, frames_per_file=126, hsv_threshold=40):
    results = defaultdict(list)

    video_manager = VideoManager(files)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=hsv_threshold))
    base_timecode = video_manager.get_base_timecode()
    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
        for i, (scene_start, scene_end) in enumerate(scene_list):
            f = scene_end.get_frames() // frames_per_file 
            scene_change = scene_end.get_frames() - frames_per_file * f
            if f < len(files) and scene_change > 0:
                results[files[f]].append(scene_change)

    finally:
        video_manager.release()
    return results

def detect_scenes(x, frames_per_file = 126, hsv_threshold = 40, chunk_size = 1000, n_processes = None):
    """
    Loads the files all in one large video and matches the resulting scene changes by the frames_per_file parameter length.
    """
    results = defaultdict(list)
    with Pool(n_processes) as p:
        r = p.map(functools.partial(process, frames_per_file=frames_per_file, hsv_threshold=hsv_threshold), [x[chunk: min(chunk + chunk_size, len(x))] for chunk in range(0, len(x), chunk_size)])
    for result in r:
        for path, scenes in result.items():
            results[path] += scenes

    return results

def detect_and_write(files, filename = "scene.changes.pkl", frames_per_file = 126, hsv_thresholds = [40], n_processes = None):
    dicts = [detect_scenes(files, frames_per_file, threshold, n_processes=n_processes) for threshold in hsv_thresholds]
    thresholds = dict(list(zip(hsv_thresholds, dicts)))
    with open(filename, "wb") as out:
        pkl.dump(thresholds, out)
    return thresholds

def main(data_dir):
    x = (mp.get_datastream(data_dir, ext=".resized.mp4")
     | mp.select_field("filename")
     | mp.as_list
    )
    return detect_and_write(x, filename = os.path.join(data_dir, "scene.changes.pkl"))


if __name__ == "__main__":
    scenes = main(data_dir)
