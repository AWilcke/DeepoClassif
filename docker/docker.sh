#!/bin/bash

nvidia-docker build -t "tf_classif" .
nvidia-docker run -v /home/arthur:/home/arthur -v /nfs:/nfs -p 6006:6006 -ti tf_classif bash
