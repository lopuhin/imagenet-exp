FROM nvidia/cuda:10.0-runtime-ubuntu18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential libffi-dev \
        python3-dev python3-pip tzdata \
        zlib1g-dev libjpeg-turbo8-dev \
        libopenblas-dev

WORKDIR /imagenet
RUN pip3 install -U pip
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl

COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    bash -c 'CC="cc -mavx" pip3 install pillow-simd==5.3.0.post0'

COPY . .
RUN pip3 install -e .
