# Homework 5

## Move to AI2024-hw5-v2 folder
cd ./AI2024-hw5-v2

## Install Necessary Packages
conda create -n hw5 python=3.11 -y
conda activate hw5
pip install -r requirements.txt

## Train model
python pacman.py

## Evaluate model
python pacman.py --eval --eval_model_path ./submissions/pacman_dqn.pt