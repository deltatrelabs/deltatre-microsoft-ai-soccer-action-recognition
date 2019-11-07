# This script will resize video files before further processing
# video_width is defined in config.py

from config import *

import mPyPl as mp
from mpyplx import *
from pipe import Pipe, where
from moviepy.editor import *

def resize_save(x,nfn):
    print("Resizing {}".format(x['filename']))
    clip, fx = x['video']
    fx.write_videofile(nfn)
    clip.close()

def load_resize(x):
    fn = x['filename']
    nfn = fn.replace('.full.mp4','.resized.mp4')
    x['filename'] = nfn
    if os.path.isfile(nfn):
        print("Loading resized {}".format(nfn))
        vc = VideoFileClip(nfn)
        return vc
    else:
        print("Resizing {}".format(fn))
        vc = VideoFileClip(fn).fx(vfx.resize, width=video_width)
        vc.write_videofile(nfn)
        return vc
    
def resize(x):
    v = VideoFileClip(x)
    vfxc = v.fx(vfx.resize, width=video_width)
    return (v, vfxc)

if __name__ == "__main__":
    (mp.get_datastream(data_dir,ext=".full.mp4")
     | where( lambda f: not os.path.isfile( f['filename'].replace(".full.mp4",".resized.mp4") ) )
     | mp.apply('filename','video', resize )
     | cachecomputex(".full.mp4",".resized.mp4",resize_save,lambda x,nx: print("Skipping {}".format(x['filename'])))
     | execute
    )

