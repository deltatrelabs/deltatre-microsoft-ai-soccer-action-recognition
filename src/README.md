# Pipeline

## Data

Creating clips dataset from full matches video
**NOTE: each full video should be ending with `.full.mp4`.

  **./create_clips.py**

  Split a full video in 5s clips, moving them in a different folder class. See more information in [./create_clips_README.md](./create_clips_README.md)

  **./resize_video.py**

  Resizes the videos to the sizes given in config.py

  - Input: *.full.mp4
  - Output: *.resized.mp4

---

## Features

### Audio Features

Features from pyAudioAnalysis:
<https://github.com/tyiannak/pyAudioAnalysis>

**./create_audiofeatures.py**

- Input: *.resized.mp4
- Output: *.audiofeatures.npy

### Visual Features

- VGG Embeddings

    Generates VGG embeddings for each frame using VGG16 pre-trained and combine all VGG embeddings for the frames in a clip to a matrix.

    **./create_vgg.py**

  - Input: *.resized.mp4
  - Output: *.vgg.npy

- Player Bounding Boxes

    **./create_bboxes.py**

    Calculates bounding boxes for the players on the field.

    Requires retinanet to be installed and the pre-trained retinanet model to be available and configured in config.py

    <https://github.com/fizyr/keras-retinanet>

  - Input: *.full.mp4
  - Output: *.boxes.pickle

- Optical Flow

  **./create_denseflow.py**

    Creating *dense* optical flow descriptors from resized videos

  - Input: *.resized.mp4
  - Output: *.optflow.npy

  **./create_fflow.py**

    *Focused* optical flow (just for player boxes)

  - Input: *.full.mp4
  - Output: *.focused.pickle

  **./focused.py**

    Class and helpers for the focused optical flow

- Player Poses

    Get pose information using `TFPoseEstimator`
    <https://github.com/ildoonet/tf-pose-estimation>

    **./create_pose.py**

  - Input: *.resized.mp4
  - Output: *.poses.pickle

- **Scene changes**

    Determine scene changes occurences in clips

    **./create_scene_change.py**

  - Input: *.resized.mp4
  - Output: scene.changes.pkl

- All together

  **./preprocess_pipeline.py**

    Putting all the pre-processing and feature extraction together

---

## Training

Scripts for training models. Other model training scripts can be found in Notebooks.

- Audio

  **./train_audio.py**

- Flow

  **./train_flow.py**

- VGG

  **./train_vgg.py**

---

## Visualization

**./render.py**

Render the output of configured models in a video, and produce an annotation JSON files (according to which models are configured in `renderconfig.json`).
