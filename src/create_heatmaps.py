# create heatmaps of players in a frame
# this generates a 2 x num_fames x num_features numpy matrix
# channel 1: the number of bounding boxes in in each X interval - (if the screen is split up into num_features intervals x-wise) - this will indicate grouping of players
# channel 2: the average size of these boxes in each X interval (this will indicate if it is a headshot and possibly also perspective)
# the input is a pickle file for each video clip with bounding boxes of each player

from config import *
from pipe import *
import mPyPl as mp
from mpyplx import *
import numpy as np
import pickle

image_width = 1280
num_features = 100
px_per_feature = image_width/num_features

def get_bbox_width(bbox):
    """Return the width of the bounding box
    :param bbox: the player bounding box [top_left x, top_left y, bottom_left x, bottom_left y]
    :return: the width of the bouding box
    #>>> get_bbox_width([23,12,35,20])
    #12
    """
    return (bbox[2] - bbox[0])

def get_bbox_center(bbox):
    """Return the center of the bounding box
    :param bbox: the player bounding box [top_left x, top_left y, bottom_left x, bottom_left y]
    :return: the center x, y of the bounding box
    #>>> get_bbox_center([23,12,35,20])
    #(29.0, 16.0)
    """
    return ((bbox[2]-bbox[0])/2+bbox[0], (bbox[3]-bbox[1])/2+bbox[1])

def generate_heatmap_for_frame(bboxes):
    """
    :param bboxes: the player bounding boxes for the frame
    :return: a histogram of number of players in each x-section of the image,
    and the average width of the boxes in that section
    """
    heatmap = np.zeros(num_features)
    total_widths = np.zeros(num_features)

    for bbox in bboxes:
        bbox_width = get_bbox_width(bbox)
        bbox_center = get_bbox_center(bbox)
        f_index = int(bbox_center[0]/px_per_feature)
        heatmap[f_index] += 1
        total_widths[f_index] += bbox_width

    avg_width = np.divide(total_widths, heatmap, out=np.zeros_like(total_widths), where=heatmap!=0)
    return (heatmap, avg_width)

def calcheatmap(x, nfn):
    pickle_name = x['filename']

    print("Creating player heatmaps for {}".format(pickle_name))
    with open(pickle_name, 'rb') as f:
        frames = pickle.load(f)
    num_frames = len(frames)

    heatmaps = np.zeros((num_frames, num_features))
    avg_widths = np.zeros((num_frames, num_features))

    for i, frame in enumerate(frames):
        (heatmap, avg_width) = generate_heatmap_for_frame(frame)
        heatmaps[i] = heatmap
        avg_widths[i] = avg_width

    f = np.concatenate((heatmaps, avg_widths)).reshape(2, num_frames, num_features)
    np.save(nfn, f)

def arrcalcheatmap(frames):
    num_frames = len(frames)
    heatmaps = np.zeros((num_frames, num_features))
    avg_widths = np.zeros((num_frames, num_features))
    for i, frame in enumerate(frames):
        (heatmap, avg_width) = generate_heatmap_for_frame(frame)
        heatmaps[i] = heatmap
        avg_widths[i] = avg_width
    f = np.concatenate((heatmaps, avg_widths)).reshape(2, num_frames, num_features)
    return f
   

if __name__ == "__main__":
    (mp.get_datastream(data_dir, ext=".boxes.pickle")
     | cachecomputex(".boxes.pickle", ".heatmaps.npy", calcheatmap, lambda x, nx: print("Skipping {}".format(x)))
     | execute
    )

