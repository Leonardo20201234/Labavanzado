#!/bin/bash

set -e  # Detiene el script si hay error
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential
sudo apt install -y maxima
python3 -m pip install --upgrade pip
pip install jax
pip install --upgrade jax
pip install numpy plotly

