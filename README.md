# ECE57000 Project - FixMatch Implementation

This repository contains the code to implement FixMatch, a semi-supervised learning method. The model is trained on the CIFAR-10 dataset, and this README provides the necessary steps to set up and run the training.

## Prerequisites

Ensure you have Python and `pip` installed, as well as access to a CUDA-enabled GPU for optimal training performance.

## Setup

1. Change to Project directory:
   ```bash
   cd ECE57000_Proj
   
2. Install the required dependencies:
   ```bash
   pip install torch torchvision
3. Create a directory for the dataset and download CIFAR-10:
   ```bash
   mkdir -p dataset
   cd dataset
   wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   tar -xzvf cifar-10-python.tar.gz
   cd ..

4. Train the model:
   ```bash
   python train.py --n-labeled 500 --batchsize 32 --n-epoches 20 --mu 5 --lr 0.003 --thr 0.9 --n-imgs-per-epoch 32768



