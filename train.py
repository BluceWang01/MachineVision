"""
Train a model on a custom dataset.
Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import os, sys
import argparse
import torch
import pandas
import numpy as np
import matplotlib as plt
import cv2 as cv
from torchvision import models
from torchvision import transforms


ROOT = "D:/Depository"

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5m.pt', help='initial weights path')


    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main():
    device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = models.resnet18(pretrained=True)
    model.eval()
    model.to(device)
    test_transform = transforms.Compose(transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        ))
    return


if __name__ == "__main__":
    opt =  parse_opt()
    main(opt)