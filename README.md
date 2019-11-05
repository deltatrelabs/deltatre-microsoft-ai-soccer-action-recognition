Action recognition in sports (soccer, shots)
============

This is the official repository for the project *Action Recognition in Sports: soccer, shots*, jointly developed by Deltatre and Microsoft. It's a prototype project, so please keep in mind that code is not cleaned up and not so much optimized. It may also not work on your machine (but should do on DSVMs equipped with GPUs).

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Exploratory Jupyter notebooks
    │
    ├── requirements.txt   <- The requirements file for reproducing the dev environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
                              Scripts to download or generate data, features for modeling,
                              train models and then use trained models to make
                              predictions

Preliminar configuration
------------

Prior to run any of the scripts / notebooks, create the following files (excluded with .gitignore), to avoid overriding personal preferences and settings.

## config.py

Create `/src/config.py` file to configure data paths and settings for data processing:

``` python
base_dir = '../data' # input data root directory
source_dir = '../data/raw' # input data directory  (where matches.json, marks.jsonl and video files are)
data_dir = '../data/processed' # where data for all ML tasks are

video_width = 256
video_size = (256,144)

retina_path = '../models/resnet50_coco_best_v2.1.0.h5'
pose_min_width = 55
pose_min_height = 48

audio_rate = 44100

scene_detection_file = "scene.changes.pkl"
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
