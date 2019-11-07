Action recognition in sports (soccer, shots)
============

This is the official repository for the project *Action Recognition in Sports: soccer, shots*, jointly developed by Deltatre and Microsoft. It's a work-in-progress prototype project, so please keep in mind that the code may contain bugs, it is not optimized and it has not been refactored yet.

Everything has been developed on Microsoft Azure DSVMs (Windows) equipped with GPUs.

Have a look at README.md and notebooks files in the project, to get more info on how things work.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The extracted clips dataset, used for model training.
    │   └── raw            <- The source original dataset (full matches and annotations).
    |
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Exploratory Jupyter notebooks
    │
    ├── requirements.txt   <- The requirements file for reproducing the dev environment
    │
    └── src                <- Source code for use in this project.
                              Scripts to generate data, features for modeling,
                              train models and then use trained models to make
                              predictions

Preliminar configuration
------------

Prior to run any of the scripts / notebooks, create the following file (excluded with .gitignore), to avoid overriding personal preferences and settings.

## config.py

Create `/src/config.py` file to configure data paths and settings for data processing:

``` python
base_dir = '../data'           # input data root directory (specify full-path)
source_dir = '../data/raw'     # input data directory  (where matches.json, marks.jsonl and video files are; specify full-path)
data_dir = '../data/processed' # where data for all ML tasks are (specify full-path)

video_width = 256
video_size = (256,144)

retina_path = '../models/resnet50_coco_best_v2.1.0.h5'
pose_min_width = 55
pose_min_height = 48

audio_rate = 44100

scene_detection_file = "scene.changes.pkl"
```

--------

License
--------

Copyright (C) 2019 Deltatre, Microsoft Corporation.

Licensed under the [MIT License](./LICENSE).

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
