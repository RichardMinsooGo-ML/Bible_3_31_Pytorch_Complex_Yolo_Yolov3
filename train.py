from terminaltables import AsciiTable

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from data_process.kitti_yolo_dataset import KittiYOLODataset
from utils.utils import *
from models.models import *

import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from eval_mAP import evaluate_mAP

from config.train_config import parse_train_configs

def main():
    # Get data configuration
    configs = parse_train_configs()
    
    print(configs.device)

    # Initiate model
    model = Darknet(configs.model_def, img_size=configs.img_size)
    model.apply(weights_init_normal)
    
    # Get data configuration
    classes = load_classes(configs.class_path)
    
    model = model.to(configs.device)
    
    # If specified we start from checkpoint
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            model.load_state_dict(torch.load(configs.pretrained_path))
            print("Trained pytorch weight loaded!")
        else:
            model.load_darknet_weights(configs.pretrained_path)
            print("Darknet weight loaded!")


    """
    idx_cnt = 0
    for name, param in model.named_parameters():
        layer_id = int(name.split('.')[1])
        print(idx_cnt,name, layer_id)
        idx_cnt += 1
    idx_cnt = 0
    for param in model.parameters():
        print(idx_cnt, param.requires_grad)
        idx_cnt += 1
    """
    
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "loss_x",
        "loss_y",
        "loss_w",
        "loss_h",
        "loss_im",
        "loss_re",
        "loss_obj",
        "loss_cls",
        "cls_acc",
        # "recall50",
        # "recall75",
        # "precision",
        "conf_obj",
        "conf_noobj",
    ]
    
    # learning rate scheduler config
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # Create dataloader
    dataset = KittiYOLODataset(cnf.root_dir, split='train', mode='TRAIN',
        folder='training', data_aug=True, multiscale=configs.multiscale_training)

    dataloader = DataLoader(dataset, configs.batch_size, shuffle=True,
        num_workers=configs.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)

    max_mAP = 0.0
    for epoch in range(0, configs.num_epochs, 1):

        num_iters_per_epoch = len(dataloader)        

        print(num_iters_per_epoch)

        # switch to train mode
        model.train()
        start_time = time.time()
        
        # Training        
        for batch_idx, batch_data in enumerate(tqdm.tqdm(dataloader)):
            """
            # print(batch_data)
            
            print(batch_data[0])
            print(batch_data[1])
            print(batch_data[1].shape)
            print(batch_data[2])
            
            imgs = batch_data[1]
            
            from PIL import Image
            import numpy as np

            w, h = imgs[0].shape[1], imgs[0].shape[2]
            src = imgs[0]
            # data = np.zeros((h, w, 3), dtype=np.uint8)
            # data[256, 256] = [255, 0, 0]
            
            data = np.zeros((h, w, 3), dtype=np.uint8)
            data[:,:,0] = src[0,:,:]*255
            data[:,:,1] = src[1,:,:]*255
            data[:,:,2] = src[2,:,:]*255
            # img = Image.fromarray(data, 'RGB')
            img = Image.fromarray(data)
            img.save('my_img.png')
            img.show()

            import sys
            sys.exit()
            """
            
            # data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            global_step = num_iters_per_epoch * epoch + batch_idx + 1
            
            imgs = Variable(imgs.to(configs.device))
            targets = Variable(targets.to(configs.device), requires_grad=False)

            total_loss, outputs = model(imgs, targets)
            
            # compute gradient and perform backpropagation
            total_loss.backward()

            if global_step % configs.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                # Adjust learning rate
                lr_scheduler.step()

                # zero the parameter gradients
                optimizer.zero_grad()

            # ----------------
            #   Log progress

            # ----------------
            
            # if (batch_idx+1) % len(dataloader) == 0:
            if (batch_idx+1) % int(len(dataloader)/3) == 0:

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % ((epoch+1), configs.num_epochs, (batch_idx+1), len(dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", total_loss.item())]
                    # logger.list_of_scalars_summary(tensorboard_log, global_step)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {total_loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_idx + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_idx + 1))
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

            model.seen += imgs.size(0)

        # Evaulation        
        #-------------------------------------------------------------------------------------
        
        print("\n---- Evaluating Model ----")
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate_mAP(model, configs,
            batch_size=configs.batch_size)

        val_metrics_dict = {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'AP': AP.mean(),
            'f1': f1.mean(),
            'ap_class': ap_class.mean()
        }

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, classes[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")

        max_mAP = AP.mean()
        #-------------------------------------------------------------------------------------
        # Save checkpoint
        if (epoch+1) % configs.checkpoint_freq == 0:
            torch.save(model.state_dict(), configs.save_path)
            print('save a checkpoint at {}'.format(configs.save_path))
            
            
if __name__ == '__main__':
    main()
    
"""
    parser = argparse.ArgumentParser()
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
    
    configs = parser.parse_args()
    print(configs)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""


    
