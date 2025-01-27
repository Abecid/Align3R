#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R

# echo "Starting: Tartanair"
# python datasets_preprocess/preprocess_Tartanair.py
# echo "Tartanair done"

echo "Starting: Vikitti"
python datasets_preprocess/preprocess_vikitti.py
echo "Vikitti done"