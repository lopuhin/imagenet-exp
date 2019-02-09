Imagenet Experiments
====================

Installation
------------

Download the ImageNet dataset and move validation images to labeled subfolders,
to do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
so that you have a local ``./data`` folder
with ``train`` and ``val`` sub-folders, which contain class folders::

    data
    ├── train
    │   ├── n01440764
    │   ├── n01443537
    ...
    │   └── n15075141
    └── val
        ├── n01440764
        ├── n01443537
    ...
        └── n15075141

Recommended way to run is via docker,
this requires https://github.com/NVIDIA/nvidia-docker.
But you can also install everything without Docker, check ``Dockerfile``.

Build docker image::

    docker build -t imagenet .

Example run command,
assuming imagenet data folder is in local ``./data`` folder,
and mounting code folders,
so that you don't have to re-build the image too often::

    ./run --arch resnet34 --workers 4

All runs are recorded into ``./mlruns`` folder, you can use ``./mlflow-ui``
to monitor them (use ``./mlflow-ui --host 0.0.0.0`` to expose it).
Note that 5000 port is hardcoded in ``./mlflow-ui``.
