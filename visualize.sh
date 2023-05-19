#!/usr/bin/env bash

# execute this script when you are in the main 
DATASET_DIR="$PWD/dataset/"

echo ""
echo "execute visualization on dataset in $DATASET_DIR"
echo ""

cd mv3dpose/
python3 visualize.py $DATASET_DIR