FROM tensorflow/tensorflow:latest-devel-gpu

LABEL maintainer "Arthur Wilcke <arthur@deepomatic.com>"

RUN apt update && \
    apt install -y git protobuf-compiler python-pil python-lxml

RUN pip install progressbar

VOLUME /home/arthur
VOLUME /nfs
WORKDIR /home/arthur/tf_classif

ENV PYTHONPATH $PYTHONPATH:/home/arthur/tf_classif/models:/home/arthur/tf_classif/models/slim
