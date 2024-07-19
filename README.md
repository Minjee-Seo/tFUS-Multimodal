# Multi-Modal Networks for Real-Time Monitoring of Intracranial Acoustic Field During tFUS Therapy

## Overview
This repository contains the implementation of networks designed for real-time simulation of intracranial pressure map generated during transcranial focused ultrasound (tFUS) therapy.

## Features
- Convolutional neural network (CNN) based Autoencoder and U-Net
- Swin Transformer based U-Net
- Python codes for training, evaluation, loading dataset
- Pre-trained model weights
> Note: Dataset is not provided due to privacy concerns.

## Requirements
- torch
- tqdm
- timm
- einops
- matplotlib
- numpy
- h5py
- natsort

Install all prerequisites with `pip install -r requirements.txt`

## Installation
Clone this repository: `git clone https://github.com/Minjee-Seo/tFUS-Multimodal.git`

## Pre-trained model weights
You can get following model weights from [here](https://drive.google.com/file/d/1pE-xqcxV8AOtBDFlODUCiRsf2kdIvwIs/view?usp=sharing):

- CT-trained Autoencoder (AECT.pth)
- MR-trained Autoencoder (AEMR.pth)
- CT-trained CNN U-Net (UNetCT.pth)
- MR-trained CNN U-Net (UNetMR.pth)
- CT-trained Swin U-Net (SwinCT.pth)
- MR-trained Swin U-Net (SwinMR.pth)
- Transfer learning CNN U-Net (UNetCTpMR.pth)

## Usage
1. Edit `dataset.py` to load your data.
2. Run `python train.py --your_project_name` for training.
3. Run `python test.py --your_project_name` for evaluation.

## Contact
For any queries, please reach out to [Minjee Seo](islandz@yonsei.ac.kr).
