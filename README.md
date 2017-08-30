# DeepoClassif

Repository for tensorflow classification for Deepomatic. 

It is a slightly modified version of the TF-slim(https://github.com/tensorflow/models/tree/master/slim). The tutorials there apply for this, except _make_tf_record.py_ can be used to create datasets from Deepomatic json format.
Pretrained checkpoints for fine-tuning can be found [here](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)

## Getting started

After cloning the repository, run the following commands, replacing the mounted volumes with the location of your data:
```bash
cd docker
./docker.sh \
  -d /path/to/datasets \
  -r /path/to/save/runs
```

This will build the docker with everything you need.

## Usage

First you will need to create your datasets, using the provided _make_tf_record.py_. Once it has been created, you must create a dataset provider for it, stored under _models/slim/datasets_ and added to the _dataset_factory.py_. This will allow you to simply use the dataset's name to select it when training. For examples on how to create a dataset provider, look at _compass.py_ or _vinci.py_.

Once this is done and all your data is mounted as indicated above, you are ready to start training. From the docker (which can be entered through _docker/enter_docker.sh_), simply use _models/slim/train_image_classifier.py_ and _eval_image_classifier.py_ as indicated in the documentation.
For your convenience, a demo script is provided in _demo_train.sh_ and _demo_eval.py_.
