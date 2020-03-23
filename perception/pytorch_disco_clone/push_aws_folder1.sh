#!/bin/bash

# e.g.:
# ./tb.sh 3456 logs_carla_sta

folder_name=${1}
echo "${folder_name}"
rsync -avtu --exclude="log*" --exclude="cuda_ops*"  --exclude="checkpoints" --exclude="__py**"  ./${folder_name}  aws1:/projects/repos/pytorch_disco/checkpoints/