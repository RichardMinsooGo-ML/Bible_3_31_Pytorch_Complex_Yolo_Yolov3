import os
import argparse

import torch
from easydict import EasyDict as edict
import utils.config as kitti_cfg


def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of Complex YOLOv3')
    
    
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=kitti_cfg.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    # configs = parser.parse_args()


    configs = edict(vars(parser.parse_args()))


    return configs
