import os
import argparse

import torch
from easydict import EasyDict as edict
from utils.utils import *

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of Complex YOLOv3')

    # parser.add_argument("--pretrained_path", type=str, default="checkpoints/yolov3_ckpt_epoch-298.pth", help="if specified starts from checkpoint model")
    # parser.add_argument("--pretrained_path", type=str, default="checkpoints/Complex_yolo_V3.pth", help="if specified starts from checkpoint model")
    
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/Complex_yolo_yolo_v3.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--save_path", type=str, default="checkpoints/Complex_yolo_yolo_v3.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--class_path", type=str,   default="dataset/classes.names", help="path to class label file")
    
    parser.add_argument("--num_epochs"  , type=int, default=2, help="number of epochs")
    parser.add_argument("--batch_size"  , type=int, default=2, help="size of each image batch")
    
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--checkpoint_freq", type=int, default=2, metavar='N', help='frequency of saving checkpoints (default: 2)')
    
    configs = edict(vars(parser.parse_args()))

    configs.iou_thres  = 0.5
    configs.conf_thres = 0.5
    configs.nms_thres  = 0.5
                
    ############## Dataset, logs, Checkpoints dir ######################
    
    configs.dataset_dir = os.path.join('dataset', 'kitti')
    configs.ckpt_dir    = 'checkpoints'
    configs.logs_dir    = 'logs'

    if not os.path.isdir(configs.ckpt_dir):
        os.makedirs(configs.ckpt_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    ############## Hardware configurations #############################    
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    return configs
