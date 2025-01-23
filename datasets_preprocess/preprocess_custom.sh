#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R

echo "Starting: Spring"
python datasets_preprocess/preprocess_Spring.py
echo "Spring done"