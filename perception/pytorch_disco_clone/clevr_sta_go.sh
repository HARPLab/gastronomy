#!/bin/bash

set -e # exit on error

echo "-----"
echo "CLEVR_STA GO"
echo "-----"

MODE="CLEVR_STA"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CLEVR_STA GO DONE"
echo "----------"

