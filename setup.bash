#!/bin/bash
set -e  # Exit on error


# Create and activate virtual environment
echo "Creating Python virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip==23.0.1

# Install Python packages
echo "Installing Python packages..."
pip install \
    numpy \
    scipy \
    ipython \
    jupyterlab \
    cython \
    diskcache \
    torch \
    torchvision \
    gpustat \
    gdown

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run: source .venv/bin/activate" 
