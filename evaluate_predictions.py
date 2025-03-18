import os
import numpy as np
# Local import
from tqdm import tqdm 
import sklearn.metrics as skmetrics 
import configs.eval_config as cfg 
from PIL import Image 
import json 

def convert_multichannel_mask(mask):
    """
    Converts a 3-channel mask values 0 or 255 at each channel into a 
    single-channel labeled mask with labels 0, 1, 2, 3 corresponding to each channel.
    
    Parameters:
        mask : 3-channel mask with values 0 or 255 at each channel
    """

    # Initialize output mask with zeros (background)
    output_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Assign class labels, based on the value 255 in each channel
    output_mask[mask[:, :, 0] == 255] = 1  
    output_mask[mask[:, :, 1] == 255] = 2  
    output_mask[mask[:, :, 2] == 255] = 3  

    return output_mask

def compute_scores_for_probabilistic_prediction(gt_bin, pred_prob):
    """
    Finds the threshold with the best F1 score and returns the best F1, precision, and recall scores.
    Parameters:
        gt_bin : binary ground truth mask
        pred_prob : probabilistic prediction of the sclera class for each pixel 
    """
    with np.errstate(invalid='ignore', divide='ignore'):  # Ignore division by zero as it's handled below
        # Compute P/R curve of probabilistic prediction
        precisions, recalls, thresholds = skmetrics.precision_recall_curve(gt_bin, pred_prob, pos_label=1)
    
    thresholds = np.append(thresholds, 1.)

    # Hack for edge cases (delete points with the same recall - this also deletes any points with precision=0, recall=0)
    recalls[~np.isfinite(recalls)] = 0  # division by zero in above P/R curve should result in 0
    # Get duplicate indices
    idx_sort = np.argsort(recalls)
    sorted_recalls_array = recalls[idx_sort]
    vals, idx_start, count = np.unique(sorted_recalls_array, return_counts=True, return_index=True)
    
    duplicates = list(filter(lambda x: x.size > 1, np.split(idx_sort, idx_start[1:])))
    if duplicates:
        # We need to delete everything but the one with maximum precision value
        for i, duplicate in enumerate(duplicates):
            duplicates[i] = sorted(duplicate, key=lambda idx: precisions[idx])[:-1]
        to_delete = np.concatenate(duplicates)
        recalls = np.delete(recalls, to_delete)
        precisions = np.delete(precisions, to_delete)
        thresholds = np.delete(thresholds, to_delete)

    # Find threshold with the best F1-score and update scores at this index
    f1scores = 2 * precisions * recalls / (precisions + recalls)
    idx = f1scores.argmax()

    return f1scores[idx], precisions[idx], recalls[idx]

def main():
    print("Evaluate predictions...")

    for val_folder in cfg.which_val_folders:
        gt_folder_path = os.path.join(cfg.data_dir , val_folder, "Masks")
        gt_files = os.listdir(gt_folder_path) 

        pred_folder_path = os.path.join(cfg.pred_folder, cfg.which_train_folder, val_folder)
        
        results = {
            "Binary": {"F1":[], "IoU":[]}, 
            "Probabilistic": {"F1":[], "Precision":[], "Recall":[]},
        }

        for img_name in tqdm(gt_files): 
            gt_path = os.path.join(gt_folder_path, img_name)
            gt_bin = Image.open(gt_path).convert("L")
            gt_bin = np.asarray(gt_bin).ravel() // 255
            
            # Compute metrics on binary predictions of sclera 
            pred_bin_path = os.path.join(pred_folder_path, "Binarised" , img_name)
            pred_bin = Image.open(pred_bin_path).convert('L')
            pred_bin = np.asarray(pred_bin).ravel() // 255
            
            f1_bin = skmetrics.f1_score(gt_bin, pred_bin, average="macro", labels=[0, 1])
            iou_bin =skmetrics.jaccard_score(gt_bin, pred_bin)
            results["Binary"]["F1"].append(f1_bin)
            results["Binary"]["IoU"].append(iou_bin)
            
            # Compute metrics on probabilistic predictions of sclera
            pred_prob_path = os.path.join(pred_folder_path, "Predictions", img_name)
            pred_prob = np.asarray(Image.open(pred_prob_path).convert('L'))
            pred_prob = np.asarray(pred_prob).ravel() / 255
            
            f1_prob, precision_prob, recall_prob = compute_scores_for_probabilistic_prediction(gt_bin, pred_prob)
            results["Probabilistic"]["F1"].append(f1_prob)
            results["Probabilistic"]["Precision"].append(precision_prob)
            results["Probabilistic"]["Recall"].append(recall_prob)

        for key in results.keys():
            for metric in results[key].keys():
                results[key][metric] = np.mean(np.array(results[key][metric])) 
        
        print(results)
        # Save results dict to json 
        os.makedirs(cfg.results_folder, exist_ok=True)
        with open(f"{cfg.results_folder}/{val_folder}.json", 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

if __name__ == '__main__':
    main()