import sys,os
from config import *
import mPyPl as mp
from mPyPl.utils.image import imprint_scale
from mpyplx import *
from pipe import Pipe
from moviepy.editor import *
import numpy as np
import itertools
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import keras
import json
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
import argparse
import datetime
from create_bboxes import *
from create_heatmaps import *
from fastai_inference import instantiate_model, run_inference

def figure_json(fn):
    "Figure JSON filename from video name"
    if fn[-4:].lower()=='.mp4':
        return fn[:-4]+'.json'
    else:
        return fn+'.json'

def figure_temp(fn):
    "Figure temp filename from video name"
    if fn[-4:].lower()=='.mp4':
        return fn[:-4]+'-temp.mp4'
    else:
        return fn+'.temp.mp4'

parser = argparse.ArgumentParser(description="DeltaTre Pipeline Renderer")
parser.add_argument("fin",help="input file name",type=str,metavar='in')
parser.add_argument("out",help="output file name",type=str)
parser.add_argument("--limit",type=int,help="limit the number of frames to process",required=False)
parser.add_argument("--resize",type=int,help="resize output video to the given height",required=False)
parser.add_argument("--json",help="Output for the json file",required=False)
parser.add_argument("--log",help="Specify optional log file to use",required=False)
parser.add_argument("--temp-video",help="Temporary video filename to use",required=False)
parser.add_argument("--inspect",help="Inspect the shapes of mpypl pipeline",required=False,action='store_const',const=True, default=False)
parser.add_argument("--keep-temp", help="Keep the temp video", required=False,action='store_const',const=True, default=False)
args = parser.parse_args()

input_video = args.fin
output_video = args.out
output_json = args.json or figure_json(output_video)
temp_video = args.temp_video or figure_temp(output_video)
frame_no = 126

def get_vgg(video):
    video = np.array(video)
    #print(video.shape)
    res = vgg.predict(keras.applications.vgg16.preprocess_input(video))
    return res

def imprint(args):
    frame,z = args
    return imprint_scale(frame,z)

def log(s):
    if args.log is not None:
        with open(args.log,'a') as f:
            f.write("{} [RNDR] {}\n".format(datetime.datetime.now(),s))

print("DeltaTre Video Model Renderer")

print(" + Loading base models")
log("load vgg+retina models")
vgg = keras.applications.vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(video_size[1],video_size[0],3))
retina = models.load_model(retina_path, backbone_name='resnet50')

print(" + Opening video file")
log("open file {}".format(input_video))
vi = VideoFileClip(input_video)
no_frames = vi.fps * vi.duration
print(" + Featurizing audio")
log("start featurizing audio for {}".format(input_video))
snd_sampling = 1.0/vi.fps*audio_rate
snd_arr = vi.audio.to_soundarray() if args.limit is None else vi.audio.subclip(t_start=0,t_end=args.limit/25.0+1).to_soundarray()
audio = audioBasicIO.stereo2mono(snd_arr)
audio_features = audioFeatureExtraction.stFeatureExtraction(audio, audio_rate, 2.0*snd_sampling, snd_sampling)[0]

with open("renderconfig.json","r") as f:
    renderconfig = json.load(f)

for m in renderconfig["models"]:
    print("Loading model {}".format(m["name"]))
    log("load model {}".format(m["name"]))
    if m["type"].endswith("torch"):
        m["model"] = instantiate_model(m["path"])
        m["score_class"] = [0] if "score_class" not in m else m["score_class"]
    else:
        m["model"] = keras.models.load_model(m["path"])

    if "minimax" in m.keys():
        m["mmax"] = np.array(json.load(open(m["minimax"],"r")))
        m["min"] = m["mmax"][:,0]
        m["wid"] = m["mmax"][:,1]-m["min"]

model_names = [ m["name"] for m in renderconfig["models"]]

def apply_models(args):
    frames,vgg,frame, au, boxes, frame112 = args
    res = []
    for m in renderconfig["models"]:
        if m["type"] == "interval_vgg":
            dim = (frame_no,)+m["model"].input_shape[2:]
            res.append(float(m["model"].predict(np.expand_dims(vgg.reshape(dim), axis=0))[0][0]))
        elif m["type"] == "audio":
            a = ((au-m["min"])/m["wid"]).transpose()
            res.append(float(m["model"].predict(np.expand_dims(np.expand_dims(a, axis=2),axis=0))[0][0]))
        elif m["type"] == "frame_torch":
            idx = np.ceil(len(frames) / 2.0)
            score = run_inference(m["model"], frames[int(idx)], m["score_class"])
            res.append(float(score.flat[0]))
        elif m["type"] == "lr_frame_torch":
            left, right = (0, 1)
            idx = np.ceil(len(frames) / 2.0)
            scores = np.around(run_inference(m["model"], frames[int(idx)], m["score_class"]), decimals=4)
            # leave the bar at 0.5, substract if it's left, add if it's right
            scores = scores * 0.5
            score = -scores.flat[left] if np.argmax(scores) == left else scores.flat[right]
            res.append(0.5 + float(score))        
        elif m["type"] == "3dcnn":
            res.append(float(m["model"].predict(np.expand_dims(frame112[::8], axis=0))[0][0]))
        elif m["type"] == "box5":
            if boxes[0][0,0]>=0:
                heatmaps = boxes[::5]
                heatmaps[:,0,:]/=10.0
                heatmaps[:,1,:]/=1280.0
                x = float(m["model"].predict(np.einsum("ikj->ijk",heatmaps).reshape(1,26,10,10,2))[0])
                m["last"] = x
            res.append(m["last"] or 0.0)
        else:
            res.append(0.0)
    return res

first_writeout = True
def write_out(args):
    global first_writeout
    no,scores = args
    d = { "no" : int(no[60]), "scores" : [ { "model" : m, "score": s} for m,s in zip(model_names,scores)] }
    with open(output_json,"w" if first_writeout else "a") as f:
        if first_writeout:
            f.write("[\n")
            first_writeout=False
        else:
            f.write(',\n')
        f.write(json.dumps(d))

@Pipe
def as_fields(datastream,field_names):
    """
    Convert stream of any objects into proper datastream of `mdict`'s, with named fields
    """
    return datastream | mp.select( lambda x : mp.mdict( dict(zip(field_names,x))))

@Pipe
def mpif(datastream,expr,pipe):
    if expr:
        return datastream | pipe
    else:
        return datastream

no_frames = args.limit or no_frames

print(" + Start main rendering, {} frames...".format(no_frames))
log("start main rendering, {} frames".format(no_frames))

z_array = -1*np.ones((2,100))

frames = (
   zip(vi.iter_frames(),audio_features.transpose())
  | as_fields(['oframe','audio'])
  | mp.fenumerate("no")
  | mp.apply('oframe','frame',lambda x: mp.utils.image.im_resize(x,video_size))
  | mp.apply('oframe','frame112',lambda x: mp.utils.image.im_resize_pad(x,(112,112)))
  | mp.apply(['no','oframe'],'boxes',lambda x: z_array if x[0]%5>0 else np.array(generate_heatmap_for_frame(get_boxes(retina,x[1]))))
  | mpif(args.limit is not None, mp.take(args.limit))
  | mp.apply('no','nox',lambda x: np.array([x]))
  | apply_batch('frame','vgg',get_vgg,batch_size=32)
  | silly_progress(elements=no_frames)
  | sliding_window_npy(['nox','oframe','frame','vgg','audio','boxes','frame112'], size=126, stride=25) # using stride of 1s = 25 frames
  | mp.apply('oframe', 'midframe', lambda x: x[60])
  | mp.apply('midframe','midframe_res',lambda x: x if args.resize is None else mp.utils.image.im_resize(x['midframe'],(None,args.resize)))
  | mp.apply(['frame','vgg','midframe','audio','boxes','frame112'],'scores',apply_models)
  | mpif(args.inspect==True,inspect)
  | mp.apply(['midframe_res','scores'],'fframe',imprint)
  | mp.iter(['nox','scores'],write_out)
  | mp.select_field('fframe')
  | mp.collect_video(temp_video)
)

## Close out the json file
with open(output_json, "a") as f:
    f.write(']\n')

print(" + Adding audio")
log("adding audio")

new_vi = VideoFileClip(temp_video)
au = vi.audio.subclip(t_start=2.5,t_end=2.5+(no_frames/25.0))
new_vi.set_audio(au)
log("writing output file {}".format(output_video))
new_vi.write_videofile(output_video)
if not args.keep_temp: os.remove(temp_video)
log("done processing {}->{}".format(input_video,output_video))
