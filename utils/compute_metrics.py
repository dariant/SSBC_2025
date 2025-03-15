import numpy as np 

def compute_iou(pred, target, which_classes = [0,1,2,3]):
    """ Custom function to compute IoU during training
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Go across all classes
    for cls in which_classes:  
        pred_inds = pred == cls #256 x 256 True or False 
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            iou = float(intersection) / float(max(union, 1))
            ious.append(iou)    
    
    return np.array(ious)



def compute_total_error(pred, target, which_classes = [0,1,2,3]):
    pred = pred.view(-1)
    target = target.view(-1)
    
    all_true = pred == target
    num_true = all_true.long().sum().data.cpu().item()
    num_false = len(target) - num_true 

    total_error = num_false / len(target)
    
    return total_error

