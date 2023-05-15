#!/usr/bin/env bash

# execute this script when you are in the main 
DATASET_DIR="$PWD/dataset/"

echo ""
echo "execute mvpose on dataset in $DATASET_DIR"
echo ""

cd mv3dpose/
python3 mvpose.py $DATASET_DIR
