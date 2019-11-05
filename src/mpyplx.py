
# Additional functions that will eventually end up in mPyPl

import json
import mPyPl as mp
from pipe import Pipe
import os,sys
from moviepy.editor import VideoFileClip
import numpy as np
import random
import cv2

def to_mdict(x):
    m = mp.mdict()
    for z in x.keys():
        m[z] = x[z]
    return m

def from_json(fn):
    with open(fn,'r') as f:
        res = json.load(f)
    for x in res:
        yield to_mdict(x)

@Pipe
def scan(l,field_name,new_field_name,func,init_state):
    """
    Perform scan (cumulitive sum) of the datastream, using given function `func` with initial state `init_state`.
    Results are places into `new_field_name` field.
    :param l: datastream
    :param field_name: field name (or list of names) to use
    :param new_field_name: field name to use for storing results
    :param func: fold function that takes field(s) value and state and returns state. If field_name is None, func
    accepts the whole `mdict` as first parameter
    :param init_state: initial state
    :return: final state of the fold
    """
    s = init_state
    for x in l:
        s = func(__fextract(x,field_name),s)
        x[new_field_name]=s
        yield x

@Pipe
def stratify_sample(seq,n=None,shuffle=False,field_name='class_id'):
    """
    Returns stratified samples of size `n` from each class (given dy `field_name`) in round robin manner.
    NB: This operation is cachy (caches all data in memory)
    :param l: input pipe generator
    :param n: number of samples or `None` (in which case the min number of elements is used)
    :param shuffle: perform random shuffling of samples
    :param field_name: name of field that specifies classes. `class_no` by default.
    :return: result sequence
    """
    data = {}
    for x in seq:
        t = x[field_name]
        if t not in data.keys(): data[t] = []
        data[t].append(x)
    if n is None:
        n = min([len(data[t]) for t in data.keys()])
    else:
        n = min(n,min([len(data[t]) for t in data.keys()]))
    if shuffle:
        for t in data.keys(): random.shuffle(data[t])
        for i in range(n):
            for t in data.keys():
                yield data[t][i]

@Pipe
def stratify_sample_tt(seq,n_samples=None,shuffle=False,class_field_name='class_id',split_field_name='split'):
    """
    Returns stratified training, test and validation samples of size `n_sample` from each class
    (given dy `class_field_name`) in round robin manner.
    `n_samples` is a dict specifying number of samples for each split type (or None).
    NB: This operation is cachy (caches all data in memory)
    :param l: input pipe generator
    :param n_samples: dict specifying number of samples for each split type or `None` (in which case the min number of elements is used)
    :param shuffle: perform random shuffling of samples
    :param class_field_name: name of field that specifies classes. `class_id` by default.
    :param split_field_name: name of field that specifies split. `split` by default.
    :return: result sequence
    """
    if n_samples is None: n_samples = {}
    data = {}
    for x in seq:
        t = x[class_field_name]
        s = x[split_field_name]
        if s not in data.keys(): data[s] = {}
        if t not in data[s].keys(): data[s][t] = []
        data[s][t].append(x)
    for s in data.keys(): # TODO: make sure train data is returned first
        if n_samples.get(s,None) is None:
            n = min([len(data[s][t]) for t in data[s].keys()])
        else:
            n = min(n_samples.get(s),min([len(data[s][t]) for t in data[s].keys()]))
        if shuffle:
            for t in data[s].keys(): random.shuffle(data[s][t])
        for i in range(n):
            for t in data[s].keys():
                yield data[s][t][i]

@Pipe
def sliding_window_npy(seq,field_names,size,stride=1,cache=10):
    """
    Create a stream of sliding windows from a given stream.
    :param seq: Input sequence
    :param field_names: Field names to accumulate
    :param size: Size of sliding window
    :param cache: Size of the caching array, in a number of `size`-chunks.
    :return: mPyPl sequence containing numpy arrays for specified fields
    """
    cachesize = size*cache
    buffer = None
    n = 0
    for x in seq:
        data = { i : x[i] for i in field_names }
        if n==0:
            buffer = { i : np.zeros((cachesize,)+data[i].shape) for i in field_names }
        if n<cachesize: # fill mode
            for i in field_names: buffer[i][n] = data[i]
            n+=1
        else: # spit out mode
            for i in range(0, cachesize-size, stride):
                yield mp.mdict( { j : buffer[j][i:i+size] for j in field_names })
            for i in field_names: buffer[i] = np.roll(buffer[i],(1-n)*size,axis=0)
            n=size
    # spit out the rest
    if n>size:
        for i in range(0, n-size, stride):
            yield mp.mdict( { j : buffer[j][i:i+size] for j in field_names })

def mdinspect(md,message="Inspecting mdict"):
    flds = md.keys()
    print(message)
    for x in flds:
        add = ""
        t = md.get(x,None)
        if t is not None:
            add = ", type={}".format(type(t))
            if type(t) is np.ndarray:
                add +=", shape={}".format(t.shape)
        print(" + {}, eval strategy={}{}".format(x,md.eval_strategies.get(x,None),add)) # TODO: print eval options and type

@Pipe
def inspect(seq,func=None,message="Inspecting mdict"):
    """
    Print out the info about the fields in a given stream
    :return: Original sequence
    """
    f = True
    for x in seq:
        if f:
            f=False
            if func is not None: func(x)
            else:
                mdinspect(x,message=message)
        yield x


@Pipe
def summarize(seq,field_name,func=None,msg=None):
    """
    Compute a summary of a given field (eg. count of different values). Resulting dictionary is either passed to `func`,
    or printed on screen (if `func is None`).
    :param seq: Datastream
    :param field_name: Field name to summarize
    :param func: Function to call after summary is obtained (which is after all stream processing). If `None`, summary is printed on screen.
    :param msg: Optional message to print before summary
    :return: Original stream
    """
    if field_name is None:
        return seq
    d = {}
    for x in seq:
        c = x.get(field_name)
        if c is not None:
            d[c] = d.get(c,0)+1
        yield x
    if func is not None:
        func(d)
    else:
        if len(d.keys())>0:
            if msg is not None: print(msg)
            for t in d.keys():
                print(" + {}: {}".format(t,d[t]))

@Pipe
def summary(seq,class_field_name='class_name',split_field_name='split'):
    """
    Print a summary of a data stream
    :param seq: Datastream
    :param class_field_name: Field name to differentiate between classes
    :param split_field_name: Field name to indicate train/test split
    :return: Original stream
    """
    return seq | summarize(field_name=class_field_name,msg="Classes:") | summarize(field_name=split_field_name,msg="Split:")

@Pipe
def cachecompute(seq,orig_ext,new_ext,func_yes=None,func_no=None):
    """
    Given a sequence of filenames with extension `orig_ext`, compute and save new files with extension `new_ext`.
    If target file does not exist, `func_yes` is executed (which should produce that file), otherwise `func_no`.
    Results of `func_yes` and `func_no` are returned.
    """
    for x in seq:
        nx = x.replace(orig_ext,new_ext)
        if not os.path.isfile(nx):
            if func_yes: yield func_yes(x,nx)
        else:
            if func_no: yield func_no(x,nx)

@Pipe
def cachecomputex(seq,orig_ext,new_ext,func_yes=None,func_no=None,filename_field='filename'):
    """
    Given a sequence with filenames in `filename_field` field with extension `orig_ext`, compute and save new files with extension `new_ext`.
    If target file does not exist, `func_yes` is executed (which should produce that file), otherwise `func_no`.
    Original sequence is returned. `func_yes` and `func_no` are passed the original `mdict` and new file name.
    """
    for x in seq:
        fn = x[filename_field]
        nfn = fn.replace(orig_ext,new_ext)
        if not os.path.isfile(nfn):
            if func_yes: func_yes(x,nfn)
        else:
            if func_no: func_no(x,nfn)
        yield x


@Pipe
def execute(l):
    """
    Runs all elements of the pipeline, ignoring the result
    The same as _ = pipe | as_list
    :param l: Pipeline to execute
    """
    list(l)

@Pipe
def datasplit_by_pattern(datastream,train_pattern=None,valid_pattern=None,test_pattern=None):
    """
    Attach data split info to the stream according to some pattern in filename.
    :param datastream: Datastream, which should contain the field 'filename', or be string stream
    :param train_pattern: Train pattern to use. If None, all are considered Train by default
    :param valid_pattern: Validation pattern to use. If None, there will be no validation.
    :param test_pattern: Test pattern to use. If None, there will be no validation.
    :return: Datastream augmented with `split` field
    """
    def check(t,pattern):
        if pattern is None:
            return False
        if isinstance(pattern,list):
            for x in pattern:
                if x in t:
                    return True
            return False
        else:
            return pattern in t

    def mksplit(x):
        t = os.path.basename(x['filename']) if 'filename' in x.keys() else x
        if check(t,valid_pattern):
            return mp.SplitType.Validation
        if check(t,test_pattern):
            return mp.SplitType.Test
        if train_pattern is None:
            return mp.SplitType.Train
        else:
            if check(t,train_pattern):
                return mp.SplitType.Train
            else:
                return mp.SplitType.Unknown

    return datastream | mp.fapply('split',mksplit)

@Pipe
def apply_nx(datastream, src_field, dst_field, func,eval_strategy=None,print_exceptions=False):
    """
    Same as `apply`, but ignores exceptions.
    """
    def applier(x):
        r = (lambda : __fnapply(x,src_field,func)) if mp.lazy_strategy(eval_strategy) else __fnapply(x,src_field,func)
        if dst_field is not None and dst_field!='':
            x[dst_field]=r
            if eval_strategy:
                x.set_eval_strategy(dst_field,eval_strategy)
        return x
    for x in datastream:
        try:
            yield applier(x)
        except:
            if print_exceptions:
                print("[mPyPl] Exception: {}".format(sys.exc_info()))
            pass

@Pipe
def apply_batch(datastream, src_field, dst_field, func, batch_size=32):
    """
    Apply function to the field in batches. `batch_size` elements are accumulated into the list, and `func` is called
    with this parameter.
    """
    n=0
    arg=[]
    seq=[]
    for x in datastream:
        f = __fextract(x,src_field)
        arg.append(f)
        seq.append(x)
        n+=1
        if (n>=batch_size):
            res = func(arg)
            for u,v in zip(seq,res):
                u[dst_field] = v
                yield u
            n=0
            arg=[]
            seq=[]
    if n>0: # flush remaining items
        res = func(arg)
        for u, v in zip(seq, res):
            u[dst_field] = v
            yield u


@Pipe
def batch(datastream,k,n):
    """
    Separate only part of the stream for parallel batch processing. If you have `n` nodes, pass number of current node
    as `k` (from 0 to n-1), and it will pass only part of the stream to be processed by that node. Namely, for i-th
    element of the stream, it is passed through if i%n==k
    :param datastream: datastream
    :param k: number of current node in cluster
    :param n: total number of nodes
    :return: resulting datastream which is subset of the original one
    """
    i=0
    for x in datastream:
        if i%n==k:
            yield x
        i+=1

## Image utils

def imprint(args,width=10,sep=3,offset=10,colors=[((255,0,0),(0,255,0))]):
    """
    A function used to imprint the results of the model into an image / video frame.
    :param args: A list of input values. First value should be an image, the rest are values in the range 0..1.
    :return: Image with imprinted indicators
    """
    frame, *scores = args
    h = frame.shape[0]
    for i,z in enumerate(scores):
        lc = i%len(colors)
        clr = colors[lc][0] if z<0.5 else colors[lc][1]
        off = offset+i*(width+sep)
        cv2.rectangle(frame,(off,offset),(off+width,h-offset),clr,1)
        cv2.line(frame,(off+width//2,h-offset),(off+width//2,h-offset-int((h-2*offset)*z)),clr,width-1)
    return frame


## Moviepy-specific
@Pipe
def close_moviepy_video(seq,video_field='video'):
    """
    Close all readers associated with open moviepy clip stored in `video_filename` field
    :param seq: Input data stream
    :param video_filename: Field with moviepy clip (detaults to `video`)
    :return: original datastream without video field
    """
    for x in seq:
        clp = x[video_field]
        clp.reader.close()
        clp.audio.reader.close_proc()
        del x[video_field]
        yield x

@Pipe
def load_moviepy_video(seq,filename_field='filename',video_field='video',eval_strategy=mp.EvalStrategies.LazyMemoized):
    return seq | mp.apply(filename_field,video_field,lambda x: VideoFileClip(x),eval_strategy=eval_strategy)


@Pipe
def silly_progress(seq,n=None,elements=None,symbol='.',width=40):
    """
    Print dots to indicate that something good is happening. A dot is printed every `n` items.
    :param seq: original sequence
    :param n: number of items to process between printing a dot
    :param symbol: symbol to print
    :return: original sequence
    """
    if n is None:
        n = elements//width if elements is not None else 10
    i=n
    for x in seq:
        i-=1
        yield x
        if i==0:
            print(symbol, end='')
            i=n

@Pipe
def delay(seq,field_name,delayed_field_name):
    """
    Create another field `delayed_field_name` from `field_name` that is one step delayed
    :param seq: Sequence
    :param field_name: Original existing field name
    :param delayed_field_name: New field name to hold the delayed value
    :return: New sequence
    """
    n = None
    for x in seq:
        if n is not None:
            x[delayed_field_name]=n[field_name]
            yield x
        n = x

@Pipe
def normalize_npy_value(seq,field_name,interval=(0,1)):
    """
    Normalize values of the field specified by `field_name` in the given `interval`
    Normalization is applied invividually to each sequence element
    :param seq: Input datastream
    :param field_name: Field name
    :param interval: Interval (default to (0,1))
    :return: Datastream with a field normalized
    """
    return seq | mp.sapply(field_name,lambda x: normalize_npy(x,interval))


### HotFixes

@Pipe
def as_batch(flow, feature_field_name='features', label_field_name='label', batchsize=16):
    """
    Split input datastream into a sequence of batches suitable for keras training.
    :param flow: input datastream
    :param feature_field_name: feature field name to use. can be string or list of strings (for multiple arguments). Defaults to `features`
    :param label_field_name: Label field name. Defaults to `label`
    :param batchsize: batch size. Defaults to 16.
    :return: sequence of batches that can be passed to `flow_generator` or similar function in keras
    """
    #TODO: Test this function on multiple inputs!
    batch = labels = None
    while (True):
        for i in range(batchsize):
            data = next(flow)
            # explicitly compute all fields - this is needed for all fields to be computed only once for on-demand evaluation
            flds = { i : data[i] for i in (feature_field_name if isinstance(feature_field_name, list) else [feature_field_name])}
            if batch is None:
                if isinstance(feature_field_name, list):
                    batch = [np.zeros((batchsize,)+flds[i].shape) for i in feature_field_name]
                else:
                    batch = np.zeros((batchsize,)+flds[feature_field_name].shape)
                labels = np.zeros((batchsize,1))
            if isinstance(feature_field_name, list):
                for j,n in enumerate(feature_field_name):
                    batch[j][i] = flds[n]
            else:
                batch[i] = flds[feature_field_name]
            labels[i] = data[label_field_name]
        yield (batch, labels)
        batch = labels = None

### Functions that will go into utils

def print_decorate(msg,expr):
    """
    Helper function to make a print decoration around lambda expression. It will print `msg`, and then return
    the result of `expr`. If it typically used in expressions like
    `lambda x: print_decorate("Processing {}".format(x), do_smth_with(x))`
    :param msg: Message to print
    :param expr: Expression to compute
    :return: expr
    """
    print(msg)
    return expr

import types

def enlist(x):
    """
    Make sure that the specified value is a list. If it's a list - returns it unchanged. If it's a value, returns
     a list with this value.
    :param x: input value
    :return: list
    """
    if isinstance(x, list):
        return x
    elif isinstance(x,types.GeneratorType):
        return list(x)
    else:
        return [x]

def normalize_npy(x,interval=(0,1)):
    """
    Normalize specified numpy array to be in the given inteval
    :param x: Input array
    :param interval: Interval tuple, defaults to (0,1)
    :return: Normalized array
    """
    mi = np.min(x)
    ma = np.max(x)
    return (x-mi)/(ma-mi)*(interval[1]-interval[0])+interval[0]

def unzip_list(x):
    u, v = zip(*x)
    return list(u), list(v)

#### Functions below are copied from mPyPl core and will be eventually removed
def __fextract(x,field_name):
    """
    Extract value of a given field or fields.
    :param x: mdict
    :param field_name: name of a field or list of field names
    :return: value or list of values
    """
    if field_name is None: return x
    if isinstance(field_name, list) or isinstance(field_name, np.ndarray):
        return [x[key] for key in field_name]
    else:
        return x[field_name]


def __fnapply(x,src_field,func):
    """
    Internal. Apply function `func` on the values extracted from dict `x`. If `src_fields` is string, function of one argument is expected.
    If `src_fields` is a list, `func` should take list as an argument.
    """
    return func(__fextract(x,src_field))


#### NEW SPRINT

import threading

@Pipe
def mpapply(datastream, src_field, dst_field, func, threads=6):
    """
    Same as apply, but for multi-threading
    """
    def applier(x):
        x[dst_field] = __fnapply(x,src_field,func)
        return x
    return None
