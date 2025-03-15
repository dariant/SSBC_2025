import os
import numpy as np
# Local import
from tqdm import tqdm 
from sklearn.metrics import f1_score
import configs.eval_config as cfg 
from PIL import Image 
import json 

def compute_iou(pred, target, which_classes = [0,1,2,3]):
    ious = []

    # Compute IoU for each specified class
    for cls in which_classes:  
        pred_inds = (pred == cls)  # Boolean mask for predicted class
        target_inds = (target == cls)  # Boolean mask for ground truth class
        
        intersection = np.sum(np.logical_and(pred_inds, target_inds))
        union = np.sum(pred_inds) + np.sum(target_inds) - intersection

        if union > 0:
            iou = float(intersection) / float(union)
        else:
            iou = 0.0  # Avoid division by zero        
        
        ious.append(iou)

    return np.array(ious)



def compute_total_error(pred, target):
    num_false = np.sum(pred != target) 
    total_pixels = target.size 
    total_error = num_false / total_pixels  
    return total_error

def convert_multichannel_mask(mask: np.ndarray) -> np.ndarray:
    """
    Converts a 3-channel binary mask (values 0 or 255) into a single-channel labeled mask.
    """

    # Initialize output mask with zeros (background)
    output_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Assign class labels, based on the value 255 in each channel
    output_mask[mask[:, :, 0] == 255] = 1  
    output_mask[mask[:, :, 1] == 255] = 2  
    output_mask[mask[:, :, 2] == 255] = 3  

    return output_mask

def main():
    print("Test...")
    which_classes = range(cfg.num_classes)
    
    if cfg.all_or_sclera == "sclera":
        which_classes = [0,1]

    for val_folder in cfg.which_val_folders:

    

        print("Which classes:", which_classes)    
        
        gt_folder_path = os.path.join(cfg.data_dir , val_folder, "labels")
        gt_files = os.listdir(gt_folder_path) 

        pred_folder_path = os.path.join(cfg.pred_folder, cfg.which_train_folder, val_folder)

        
        all_IoU = []; all_f1 = []; all_total_errors = []

        for img_name in tqdm(gt_files): 
            label_path = os.path.join(gt_folder_path, img_name)
            pred_path = os.path.join(pred_folder_path, img_name)

            pred = np.asarray(Image.open(pred_path))
            label = np.asarray(Image.open(label_path))
            
            pred = convert_multichannel_mask(pred)
            label = convert_multichannel_mask(label)

            # if only sclera and periocular 
            if which_classes == [0, 1]:
                pred[pred == 2] = 0
                pred[pred == 3] = 0

            pred = pred.ravel()
            label = label.ravel()

            # print(np.unique(pred))
            # print(np.unique(label))
            # exit()
            iou_classes = compute_iou(pred, label, which_classes)
            iou_mean = iou_classes.mean()
            total_error = compute_total_error(pred, label)
            f1 = f1_score(pred, label, average="macro", labels=which_classes)
            
            all_IoU.append(iou_mean)
            all_total_errors.append(total_error)
            all_f1.append(f1)

        ious, f1s, total_errors = np.array(all_IoU), np.array(all_f1), np.array(all_total_errors)
        
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