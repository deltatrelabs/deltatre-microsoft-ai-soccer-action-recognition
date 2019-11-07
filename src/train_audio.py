# Model training on audio

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
 | mp.take(1000)
 | mp.iter('filename',lambda x: print("Processing {}".format(x)))
 | mp.apply('filename','aud',lambda x: np.load(x.replace('.resized.mp4','.audiofeatures.npy')))
 | normalize_npy_value('aud',interval=(-1,1))
 | mp.as_list
)

tr,te = data | mp.apply('aud','amean',lambda x : np.mean(x,axis=1)) | mp.make_train_test_split

def unzip(l):
    t1,t2 = zip(*l)
    return list(t1),list(t2)

X_tr, Y_tr = unzip(tr| mp.select_field(['amean','class_id']) | mp.as_list)
X_te, Y_te = unzip(te| mp.select_field(['amean','class_id']) | mp.as_list)

X_tr[0]

from sklearn import svm
from sklearn import metrics

def SVM(xtr, xte, ytr, yte, title="confusion matrix", C=10., gamma=0.001):
    print(title)
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xte)
    print(metrics.classification_report(yte, y_pred))
    print(metrics.confusion_matrix(yte, y_pred))
    # plotConfusion(yte, y_pred, title=title)
    return y_pred, clf

cls = SVM(X_tr,X_te,Y_tr,Y_te, C=25)

print(cls)

exit(0)
trainstream, valstream = data | mp.make_train_test_split

no_intervals = 200
no_features = 34
no_train = data | mp.filter('split',lambda x: x==mp.SplitType.Train) | mp.count
no_test = data | mp.filter('split',lambda x: x==mp.SplitType.Test) | mp.count
print("Training samples = {}\nTesting samples = {}".format(no_train,no_test))
batchsize=32

model = Sequential()
#model.add(Conv1D(10,5,input_shape=(no_features,no_intervals),data_format='channels_first'))
#model.add(MaxPooling1D(data_format='channels_first'))
#model.add(Conv1D(20,5,input_shape=(no_features,no_intervals),data_format='channels_first'))
#model.add(MaxPooling1D(data_format='channels_first'))
model.add(Flatten(input_shape=(no_features,no_intervals)))
model.add(Dropout(0.3))
model.add(Dense(3,activation='softmax',kernel_initializer='glorot_uniform')) #kernel_regularizer=l2(0.01)))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.03),
              metrics=['acc'])
model.summary()

history = model.fit_generator(
      trainstream | mp.infshuffle | mp.as_batch('aud', 'class_id', batchsize=batchsize),
      steps_per_epoch=no_train // batchsize,
      validation_data= valstream | mp.infshuffle | mp.as_batch('aud', 'class_id', batchsize=batchsize),
      validation_steps = no_test // batchsize,
      epochs=300)
