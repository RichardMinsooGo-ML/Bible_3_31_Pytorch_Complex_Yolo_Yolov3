
import os, sys, time, datetime, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import data_process.config as cnf
from data_process.kitti_yolo_dataset import KittiYOLODataset

from models.models import *
from utils.utils import *

def evaluate_mAP(model, configs, batch_size):
    # switch to evaluate mode
    model.eval()

    # Get dataloader
    split='valid'
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size"  , type=int  , default=4, help="size of each image batch")
    
    parser.add_argument("--model_def"   , type=str  , default="config/complex_yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--model_def"   , type=str  , default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str  , default="checkpoints/yolov3_ckpt_epoch-298.pth", help="path to weights file")
    parser.add_argument("--weights_path", type=str  , default="checkpoints/Model_complexer_yolo_V3.pth", help="path to weights file")
    
    # parser.add_argument("--weights_path", type=str  , default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
    
    parser.add_argument("--class_path"  , type=str  , default="dataset/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres"   , type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres"  , type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres"   , type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size"    , type=int  , default=cnf.BEV_WIDTH, help="size of each image dimension")
    configs = parser.parse_args()
    print(configs)


    #data_config = parse_data_config(opt.data_config)
    #class_names = load_classes(data_config["names"])
    class_names = load_classes(configs.class_path)

    # Initiate model
    model = Darknet(configs.model_def) # .to(device)
    
    # model.print_network()
    print("\n" + "___m__@@__m___" * 10 + "\n")
    
    print(configs.weights_path)
    
    assert os.path.isfile(configs.weights_path), "No file at {}".format(configs.weights_path)
    
    # Load checkpoint weights
    model.load_state_dict(torch.load(configs.weights_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configs.device = torch.device("cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx))
    model = model.to(device = device)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    model.eval()

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(model, configs, batch_size = configs.batch_size)

    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {:.4f}\n".format(AP.mean()))
