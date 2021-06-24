# python eval_mAP.py --model_def config/complex_yolov3_tiny.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v3_tiny.pth --batch_size 8
# python eval_mAP.py --model_def config/complex_yolov3.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v3.pth --batch_size 2

import numpy as np

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from data_process.kitti_yolo_dataset import KittiYOLODataset
from utils.utils import *
from models.models import *

import data_process.config as cnf
import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim


def evaluate_mAP(model, configs, batch_size):
    # switch to evaluate mode
    model.eval()

    # Get dataloader
    split='valid'
    # Create dataloader
    dataset = KittiYOLODataset(cnf.root_dir, split=split, mode='EVAL', folder='training', data_aug=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    with torch.no_grad():
        
        for batch_idx, batch_data in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            # data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] *= configs.img_size

            imgs = Variable(imgs.type(Tensor), requires_grad=False)

            # with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=configs.conf_thres, nms_thres=configs.nms_thres)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def main():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--model_def"   , type=str  , default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--pretrained_path", type=str  , default="checkpoints/Complex_yolo_yolo_v3_tiny.pth", help="path to weights file")
    
    parser.add_argument("--model_def"   , type=str  , default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_path", type=str  , default="checkpoints/Complex_yolo_yolo_v3.pth", help="path to weights file")
        
    parser.add_argument("--class_path"  , type=str  , default="dataset/classes.names", help="path to class label file")
    parser.add_argument("--batch_size"  , type=int  , default=2, help="size of each image batch")
    parser.add_argument("--iou_thres"   , type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres",  type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",   type=int,   default=cnf.BEV_WIDTH, help="size of each image dimension")
    configs = parser.parse_args()
    print(configs)
    
    ############## Hardware configurations #############################    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(configs.model_def, img_size=configs.img_size)
    
    # Get data configuration
    classes = load_classes(configs.class_path)
    
    # model.print_network()
    print("\n" + "___m__@@__m___" * 10 + "\n")
    
    print(configs.pretrained_path)    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    
    model = model.to(device)
    
    # Load checkpoint weights
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            model.load_state_dict(torch.load(configs.pretrained_path))
            print("Trained pytorch weight loaded!")
    
    # model.load_state_dict(torch.load(configs.pretrained_path))
    
    # Eval mode
    model.eval()
    

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(model, configs, batch_size = configs.batch_size)

    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                classes[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {:.4f}\n".format(AP.mean()))
            
if __name__ == '__main__':
    main()
    