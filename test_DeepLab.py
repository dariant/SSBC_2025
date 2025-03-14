from shutil import which
from matplotlib.pyplot import savefig
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import argparse
import pathlib

import numpy as np
# Local import
from utils.dataloader import DataLoaderSegmentation
from utils.deeplab_model import initialize_model
from tqdm import tqdm 
from sklearn.metrics import f1_score
import configs.test_config as cfg 
from utils.compute_metrics import compute_iou, compute_total_error
from utils.visualization import overlay_masks_on_samples, visualize_predicted_mask

def test_models(model, dataloaders, device, resume_where, which_classes=[0,1,2,3], visualise_example=False):
    print("Test model")

    all_test_IoU = []; all_test_f1 = []; all_total_errors = []

    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode
    
    # Iterate over data.
    for ind, (inputs, labels) in tqdm(enumerate(dataloaders["TEST"])):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)['out']        
        _, preds = torch.max(outputs, 1)
        
        iou_classes = compute_iou(preds, labels, which_classes)
        tmp_iou_mean = iou_classes.mean()
        total_error = compute_total_error(preds, labels, which_classes)
        
        y_pred = preds.data.cpu().numpy().ravel()
        y_true = labels.data.cpu().numpy().ravel()

        tmp_f1_score = f1_score(y_true, y_pred, average="macro", labels=which_classes)

        all_test_IoU.append(tmp_iou_mean)
        all_total_errors.append(total_error)
        all_test_f1.append(tmp_f1_score)
        
        if visualise_example:
            if ind < 10: overlay_masks_on_samples(preds, inputs, labels, resume_where, ind)
            else: break 

    # load best model weights
    return all_test_IoU, all_test_f1, all_total_errors


def main(args):
    val_dir = os.path.join(args.data_dir, args.which_val_folder)
    validate_dataset = DataLoaderSegmentation(val_dir, "val") 
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    dataloaders_dict = {"TEST": validate_dataloader}

    # Detect if GPU is available
    device = torch.device(args.device_name if torch.cuda.is_available() else "cpu")
    
    # Initialize model and load best one 
    net = initialize_model(args.num_classes, args.keep_feature_extract, use_pretrained=True)
    net.load_state_dict(torch.load(args.resume_where))
    net = net.to(device)
    net.eval() 
    
    print("Test...")

    which_classes = range(args.num_classes)
    
    if args.all_or_sclera == "sclera":
        which_classes = [0,1]

    print("Which classes:", which_classes)
        

    # Train and evaluate
    ious, f1s, total_errors = test_models(net, dataloaders_dict, device, args.resume_where, 
                                          visualise_example=args.visualise_example, which_classes=which_classes)

    ious, f1s, total_errors = np.array(ious), np.array(f1s), np.array(total_errors)
    
    print(f"IoU: {str(np.round(np.mean(ious), 3))}, ({str(np.round(np.std(ious), 3))})")
    print(f"F1: {str(np.round(np.mean(f1s), 3))}, ({str(np.round(np.std(f1s), 3))})")
    print(f"Total errors: {str(np.round(np.mean(total_errors), 3))}, ({str(np.round(np.std(total_errors), 3))})")


if __name__ == '__main__':
    main(cfg)