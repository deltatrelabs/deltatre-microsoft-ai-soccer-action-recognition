# Model training on dense optical flow

from config import *

import mPyPl as mp
import mPyPl.utils.image as mpui
from mpyplx import *
from pipe import Pipe
from moviepy.editor import *
import numpy as np
import itertools
import cv2
import math
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import *
from keras.regularizers import l2

def dense_flow_stabilize(df):
    for i,f in enumerate(df):
        vec = np.average(f,axis=(0,1))
        mask = f==0
        f = f-vec
        f[mask]=0
        df[i]=f
    return df

#first_frame = (
# mp.get_datastream(data_dir,ext=".resized.mp4")
# | mp.apply('filename','dflow',lambda x: np.load(x.replace('.resized.mp4','.optflow.npy')))
# | mp.first
#)

def flow_to_polar(flow):
    return cv2.cartToPolar(flow[..., 0], flow[..., 1])

def visualize_flow_hsv(flow):
    hsvImg = np.zeros(flow.shape[:-1]+(3,),dtype=np.uint8)
    mag, ang = flow_to_polar(flow)
    hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
    hsvImg[..., 1] = 255
    hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

# Plot the flow of first 5 frames
#flow = first_frame['dflow']
#mpui.show_images([visualize_flow_hsv(x) for x in flow[0:5]])

def safe_array(x):
    return x[np.isfinite(x)]

def get_flow_descriptor(flow,nbins=100,mag_min=0,mag_max=7):
    def descr(f,abins,mbins):
        mag, ang = flow_to_polar(f)
        h1,bins1 = np.histogram(safe_array(ang.ravel()),bins=abins)
        # print(mag.shape,mbins.shape)
        h2,bins2 = np.histogram(safe_array(mag.ravel()),bins=mbins)
        return [[x,y] for x,y in zip(h1,np.log(1+h2))] # we take log of mag histogram to make range smaller
    abins = [i*2*math.pi/nbins for i in range(0,nbins+1)]
    mbins = np.arange(nbins+1)/nbins*(mag_max-mag_min)+mag_min
    return np.array([descr(f,abins=abins,mbins=mbins) for f in flow],dtype=np.float32)

def plot_flow_descriptor(fd,step=5):
    fig,ax = plt.subplots(5,2)
    for i in range(5):
        ax[i,0].plot(fd[i*step,:,0])
        ax[i,1].plot(fd[i*step,:,1])
    plt.show()

# Plot to see how it works
# fd = get_flow_descriptor(flow)
# plot_flow_descriptor(np.log(fd))

# Get list of test videos from matches.json
test_names = (
   from_json(os.path.join(source_dir,'matches.json'))
 | mp.where(lambda x: 'Test' in x.keys() and int(x['Test'])>0)
 | mp.apply(['Id','Half'],'pattern',lambda x: "{}_{}_".format(x[0],x[1]))
 | mp.select_field('pattern')
 | mp.as_list)

data = (
 mp.get_datastream(data_dir,ext=".resized.mp4")
 | datasplit_by_pattern(test_pattern=test_names)
 | stratify_sample_tt(shuffle=True)
 | summary()
 | mp.iter('filename',lambda x: print("Processing {}".format(x)))
 | mp.apply('filename','dflow',lambda x: np.load(x.replace('.resized.mp4','.optflow.npy')),eval_strategy=mp.EvalStrategies.LazyMemoized)
 | mp.apply_npy('dflow','flowdescr', get_flow_descriptor)
 | mp.delfield('dflow')
 | mp.as_list
)

# Calculate min/max to normalize
A = np.array(data
 | mp.select_field('flowdescr')
 | mp.select(lambda x: (x[...,0].min(), x[...,0].max(), x[...,1].min(), x[...,1].max()))
 | mp.as_list)
min_ang,max_ang = min(A[:,0]),max(A[:,1])
min_amp,max_amp = min(A[:,2]),max(A[:,3])

def normalize(fd):
    fd[:,:,0] = (fd[:,:,0]-min_ang)/(max_ang-min_ang)
    fd[:, :, 1] = (fd[:, :, 1] - min_amp) / (max_amp - min_amp)
    return fd

trainstream, valstream = data | mp.apply('flowdescr','fdn', normalize) | mp.make_train_test_split

no_frames = 125
no_train = data | mp.filter('split',lambda x: x==mp.SplitType.Train) | mp.count
no_test = data | mp.filter('split',lambda x: x==mp.SplitType.Test) | mp.count
print("Training samples = {}\nTesting samples = {}".format(no_train,no_test))
batchsize=8

model = Sequential()
model.add(AveragePooling2D((2, 2),input_shape=(no_frames, 100, 2)))
model.add(Conv2D(8, (3, 3), data_format='channels_last',activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(AveragePooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(Dense(3,activation='softmax',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.1),
              metrics=['acc'])
model.summary()

history = model.fit_generator(
      trainstream | mp.infshuffle | mp.as_batch('flowdescr', 'class_id', batchsize=batchsize),
      steps_per_epoch=no_train // batchsize,
      validation_data= valstream | mp.infshuffle | mp.as_batch('flowdescr', 'class_id', batchsize=batchsize),
      validation_steps = no_test // batchsize,
      epochs=30)
