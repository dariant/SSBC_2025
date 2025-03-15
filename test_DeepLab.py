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
from utils.visualization import overlay_masks_on_samples, save_predicted_mask
import json 

def test_models(model, dataloader, device, predictions_folder, which_classes=[0,1,2,3]):
    print("Test model")

    all_test_IoU = []; all_test_f1 = []; all_total_errors = []

    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode
    
    # Iterate over data.
    for inputs, labels, img_name  in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)['out']        
        _, preds = torch.max(outputs, 1)
        
        if which_classes == [0, 1]:
            preds[preds == 2] = 0
            preds[preds == 3] = 0

        print(preds.unique())
        print(labels.unique())
        exit()        
        iou_classes = compute_iou(preds, labels, which_classes)
        tmp_iou_mean = iou_classes.mean()
        total_error = compute_total_error(preds, labels, which_classes)
        
        y_pred = preds.data.cpu().numpy().ravel()
        y_true = labels.data.cpu().numpy().ravel()

        tmp_f1_score = f1_score(y_true, y_pred, average="macro", labels=which_classes)

        all_test_IoU.append(tmp_iou_mean)
        all_total_errors.append(total_error)
        all_test_f1.append(tmp_f1_score)
        
        save_predicted_mask(preds, img_name, predictions_folder)
        
        if cfg.overlay_visualization:
            overlay_masks_on_samples(preds, inputs, labels, cfg.resume_where, img_name)
            
    # load best model weights
    return all_test_IoU, all_test_f1, all_total_errors


def main():
    
    # Detect if GPU is available
    device = torch.device(cfg.device_name if torch.cuda.is_available() else "cpu")
    
    # Initialize model and load best one 
    net = initialize_model(cfg.num_classes, cfg.keep_feature_extract, use_pretrained=True)
    net.load_state_dict(torch.load(cfg.resume_where))
    net = net.to(device)
    net.eval() 
    
    print("Test...")

    which_classes = range(cfg.num_classes)
    
    if cfg.all_or_sclera == "sclera":
        which_classes = [0,1]

    for val_folder in cfg.which_val_folders:
        val_folder_path = os.path.join(cfg.data_dir, val_folder)
        validate_dataset = DataLoaderSegmentation(val_folder_path, "val") 
        validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        predictions_folder = os.path.join(cfg.predictions_folder, f"{val_folder}_{cfg.all_or_sclera}") 
        print("Which classes:", which_classes)    
        os.makedirs(predictions_folder, exist_ok=True)

        # Train and evaluate
        ious, f1s, total_errors = test_models(net, validate_dataloader, device, predictions_folder, which_classes=which_classes)

        ious, f1s, total_errors = np.array(ious), np.array(f1s), np.array(total_errors)
        
        results_dict = {"IoU": f"{np.round(np.mean(ious), 3)} ({np.round(np.std(ious), 3)})", 
                        "F1": f"{np.round(np.mean(f1s), 3)} ({np.round(np.std(f1s), 3)})",
                        "Pixel error": f"{np.round(np.mean(total_errors), 3)} ({np.round(np.std(total_errors), 3)})"
                        }
        
        for key in results_dict.keys():
            print(key, results_dict[key])

        os.makedirs(cfg.results_folder, exist_ok=True)
        with open(f"{cfg.results_folder}/{val_folder}_{cfg.all_or_sclera}.json", 'w') as fp:
            json.dump(results_dict, fp, sort_keys=True, indent=4)

if __name__ == '__main__':
    main()