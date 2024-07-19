# Multi-Modal Networks for Real-Time Monitoring of Intracranial Acoustic Field During tFUS Therapy

## Overview
This repository contains the implementation of networks designed for real-time simulation of intracranial pressure map generated during transcranial focused ultrasound (tFUS) therapy.

## Features
- Convolutional neural network (CNN) based Autoencoder and U-Net
- Swin Transformer based U-Net
- Python codes for training, evaluation, loading dataset

## Requirements
- torch
- tqdm
- timm
- einops
- matplotlib
- numpy
- h5py
- natsort

## Installation
Clone this repository: `git clone https://github.com/Minjee-Seo/tFUS-Multimodal.git`

## Pre-trained model weights

## Usage
1. Edit `dataset.py` to load your data.
2. Run `python train.py --your_project_name` for training.
3. Run `python test.py --your_project_name` for evaluation.
