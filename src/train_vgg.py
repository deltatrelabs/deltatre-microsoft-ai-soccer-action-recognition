# Model training on vgg embeddings

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

test_names = (
   from_json(os.path.join(source_dir,'matches.json'))
 | mp.where(lambda x: 'Test' in x.keys() and int(x['Test'])>0)
 | mp.apply(['Id','Half'],'pattern',lambda x: "{}_{}_".format(x[0],x[1]))
 | mp.select_field('pattern')
 | mp.as_list)

no_frames = 126

data = (
 mp.get_datastream(data_dir,ext=".resized.mp4")
 | mp.pshuffle
 | datasplit_by_pattern(test_pattern=test_names)
 | mp.apply('filename','vgg',lambda x: np.load(x.replace('.resized.mp4','.vgg.npy')))
 | mp.apply('vgg','vggflat',lambda x: np.reshape(x,(no_frames,-1,1)))
 | mp.take(500)
 | mp.as_list
)

trainstream, valstream = data | mp.make_train_test_split

no_train = data | mp.filter('split',lambda x: x==mp.SplitType.Train) | mp.count
no_test = data | mp.filter('split',lambda x: x==mp.SplitType.Test) | mp.count
print("Training samples = {}\nTesting samples = {}".format(no_train,no_test))
batchsize=2

model = Sequential()
model.add(AveragePooling2D((10, 10),input_shape=(no_frames, 16384, 1)))
model.add(Conv2D(8, (3, 3), data_format='channels_last',activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(AveragePooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(Dense(1,activation='sigmoid',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])
model.summary()

history = model.fit_generator(
      trainstream | mp.infshuffle | mp.as_batch('vggflat', 'class_id', batchsize=batchsize),
      steps_per_epoch=no_train // batchsize,
      validation_data= valstream | mp.infshuffle | mp.as_batch('vggflat', 'class_id', batchsize=batchsize),
      validation_steps = no_test // batchsize,
      epochs=30)
