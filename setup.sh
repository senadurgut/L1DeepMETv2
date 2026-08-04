#!/bin/bash

# Create and activate new micromamba environment named 'pyg112_env' with Python 3.9
micromamba create -n pyg112_env python=3.9 -y
micromamba activate pyg112_env

# Install cudatoolkit 10.2 and PyTorch 1.12 from pytorch channel
micromamba install -c pytorch pytorch=1.12.0 cudatoolkit=10.2 -y

# Downgrade numpy to 1.24.x (last stable 1.x compatible with PyTorch 1.12)
pip install numpy==1.24.3

# Set CUDA variable for PyG wheel URLs
export CUDA="cu102"

# Install PyG dependencies pinned to torch 1.12 + cuda 10.2 wheels
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html

# Install main torch-geometric package (should match dependencies above)
pip install torch-geometric==2.6.1

# Install coffea and mplhep as requested
pip install coffea mplhep

