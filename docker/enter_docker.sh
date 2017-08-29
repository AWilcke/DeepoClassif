#!/bin/bash

DOCKER_ID=`docker ps | grep -i tf_classif | cut -d ' ' -f 1`

nvidia-docker exec -ti $DOCKER_ID bash
