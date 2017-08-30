FROM tensorflow/tensorflow:latest-devel-gpu

LABEL maintainer "Arthur Wilcke <arthur@deepomatic.com>"

RUN apt update && \
    apt install -y git protobuf-compiler python-pil python-lxml

RUN pip install progressbar

VOLUME /home
VOLUME /runs

WORKDIR /home/models

ENV PYTHONPATH $PYTHONPATH:/home/models:/home/models/slim

ENTRYPOINT ["tensorboard","--logdir","/runs"]
