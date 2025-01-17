#!/bin/bash
export PYTHONPATH=$PYTHONPATH:Align3R

echo "Starting: PointOdyssey"
python datasets_preprocess/preprocess_PointOdyssey.py
echo "PointOdyssey done"

echo "Starting: Spring"
python datasets_preprocess/preprocess_Spring.py
echo "Spring done"

echo "Starting: Tartanair"
python datasets_preprocess/preprocess_Tartanair.py
echo "Tartanair done"

echo "Starting: Vikitti"
python datasets_preprocess/preprocess_vikitti.py
echo "Vikitti done"

# python datasets_preprocess/preprocess_Flythings3D.py
# echo "Flythings3D done"

echo "Starting: Driving"
python datasets_preprocess/preprocess_Driving.py
echo "Driving done"

echo "Starting: Monkaa"
python datasets_preprocess/preprocess_Monkaa.py
echo "Monkaa done"