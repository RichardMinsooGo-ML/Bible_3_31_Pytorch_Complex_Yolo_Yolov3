import numpy as np

import os, sys, time, datetime, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import math
import cv2
import torch

from models.models import *
from utils.utils import *
import torch.utils.data as torch_data

from data_process import kitti_utils, kitti_bev_utils

from data_process.kitti_yolo_dataset import KittiYOLODataset
import data_process.config as cnf

# import utils.mayavi_viewer as mview
from utils.mayavi_viewer import show_image_with_boxes, predictions_to_kitti_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
    parser.add_argument("--model_def",  type=str,   default="config/complex_yolov3_tiny.cfg", help="path to model definition file")
    
    parser.add_argument("--class_path", type=str,   default="./dataset/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres",  type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",   type=int,   default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--save_video", type=bool,  default=True, help="Set this flag to True if you want to record video")
    # parser.add_argument("--split",      type=str,   default="valid", help="text file having image lists in dataset")
    # parser.add_argument("--folder",     type=str,   default="training", help="directory name that you downloaded all dataset")
    
    parser.add_argument("--split",      type=str,   default="detect_1", help="text file having image lists in dataset")
    parser.add_argument("--folder",     type=str,   default="detect_1", help="directory name that you downloaded all dataset")
    configs = parser.parse_args()
    print(configs)

    # Set up model
    model = Darknet(configs.model_def, img_size=configs.img_size)
    classes = load_classes(configs.class_path)
    # model.print_network()
    print("\n\n" + "-*=" * 30 + "\n\n")
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(configs.pretrained_path))
    # Eval mode
    model.eval()
    
    dataset = KittiYOLODataset(cnf.root_dir, split=configs.split, mode='TEST', folder=configs.folder, data_aug=False)
    
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if configs.save_video:
        out = cv2.VideoWriter('Detection_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (608, 791))

    start_time = time.time()
    for batch_idx, (img_paths, imgs_bev) in enumerate(data_loader):
        
        # Configure bev image
        input_imgs = Variable(imgs_bev.type(Tensor))
        outputs = model(input_imgs)

        # Get detections 
        with torch.no_grad():
            detections = non_max_suppression_rotated_bbox(outputs, configs.conf_thres, configs.nms_thres) 
        
        end_time = time.time()
        print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
        start_time = end_time

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        imgs_bev = torch.squeeze(imgs_bev).numpy()

        img_bev = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        img_bev[:, :, 2] = imgs_bev[0, :, :]  # r_map
        img_bev[:, :, 1] = imgs_bev[1, :, :]  # g_map
        img_bev[:, :, 0] = imgs_bev[2, :, :]  # b_map
        
        img_bev *= 255
        img_bev = img_bev.astype(np.uint8)
        
        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        img_rgb = cv2.imread(img_paths[0])
        calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
        img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)
        
        img_rgb = cv2.resize(img_rgb, (608,int(791-608)))
        
        # Using cv2.ROTATE_180 rotate by  
        # 180 degrees clockwise 
        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180) 
        
        out_img = np.concatenate((img_rgb, img_bev), axis=0)
        
        cv2.imshow('BEV_DETECTION_RESULT', out_img)
                
        print(img_rgb.shape)
        print(out_img.shape)
        
        if configs.save_video:
            out.write(out_img)
        
        if cv2.waitKey(0) & 0xFF == 27:
            break

    if configs.save_video:
        out.release()