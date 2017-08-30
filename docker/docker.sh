#!/bin/bash

# get flags
data_dir=''
runs_dir=''

while getopts 'd:r:' flag; do
  case "${flag}" in
    d) data_dir="${OPTARG}" ;;
    r) runs_dir="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

# get wd of main repo
PWD=`pwd | rev | cut -d / -f2- | rev`

nvidia-docker build -t "tf_classif" .
nvidia-docker run \
    -v $PWD:/home \
    -v $data_dir:/data \
    -v $runs_dir:/runs \
    -p 6006:6006 \
    -ti tf_classif
