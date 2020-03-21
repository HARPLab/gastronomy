#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_STA GO"
echo "-----"

MODE="CARLA_STA"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_STA GO DONE"
echo "----------"

