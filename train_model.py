import os
import torch
import pathlib
import numpy as np

# Local import
import random
import copy 
from tqdm import tqdm 
import configs.train_config as cfg 
from utils.deeplab_model import initialize_model
from utils.dataloader import DataLoaderSegmentation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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


def train_model(model, num_classes, dataloaders, criterion, 
                optimizer, scheduler, device, dest_dir, num_epochs=25):
    """ Function for training the model. """
    val_acc_history = []

    best_DeepLabV3_acc = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = ["IoU"]
    counter = 0
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'LR', 'train_loss', 'val_loss'] + \
        [f'train_{m}' for m in metrics] + \
        [f'val_{m}' for m in metrics]
    
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        batchsummary = {a: [0] for a in fieldnames}
        batchsummary['epoch'] = epoch

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_iou_means = []
            
            # Iterate over data.
            for inputs, labels, img_names in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Security, skip this iteration if the batch_size is 1
                if 1 == inputs.shape[0]:
                    print("Skipping iteration because batch_size = 1")
                    continue

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['out']
                    
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # Training statistics
                iou_mean = compute_iou(preds, labels, which_classes = range(num_classes))#[0,1,2,3])

                running_loss += loss.item() * inputs.size(0)
                running_iou_means.append(iou_mean.mean())
                batchsummary[f'{phase}_IoU'].append(iou_mean.mean())
                
                # Increment counter
                counter = counter + 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if running_iou_means is not None:
                epoch_acc = np.array(running_iou_means).mean()
            else:
                epoch_acc = 0.


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            batchsummary[f'{phase}_loss'] = epoch_loss

            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_DeepLabV3_acc = copy.deepcopy(model.state_dict())
                torch.save(best_DeepLabV3_acc, os.path.join(dest_dir, "best_trained_DeepLabV3.pth"))
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
            if phase == "val":
                scheduler.step(epoch_loss)
                print("Setting learning rate to:", optimizer.param_groups[0]['lr'])

            # Save current model every 5 epochs
            if 0 == epoch%5:
                current_model_path = os.path.join(dest_dir, f"checkpoint_{epoch:04}_DeepLabV3.pth")
                print(f"Save current model : {current_model_path}")
                torch.save(model.state_dict(), current_model_path)


        # end of epoch
        for field in fieldnames[4:]:
            batchsummary[field] = np.mean(batchsummary[field])


        current_learning_rate = optimizer.param_groups[0]['lr']
        batchsummary['LR'] = current_learning_rate
        print(batchsummary)

        if current_learning_rate < 0.5e-6:
            print("Early stopping, because learning rate is too low.", current_learning_rate)
            break 

    print('Best val Acc: {:4f}'.format(best_acc))

    save_path = os.path.join(dest_dir, "best_trained_DeepLabV3.pth")
    print(f"Save best model: {save_path}")
    torch.save(best_DeepLabV3_acc, save_path)
    

def main():
    set_seed(0)

    # Initialize datasets and dataloaders
    train_dir = os.path.join(cfg.data_dir, cfg.main_folder, "Training")
    val_dir = os.path.join(cfg.data_dir, cfg.main_folder, "Validation")
    
    print(f"Number of classes: {cfg.num_classes}")
    train_dataset = DataLoaderSegmentation(train_dir,"train", num_classes=cfg.num_classes ) 
    validate_dataset = DataLoaderSegmentation(val_dir, "val", num_classes=cfg.num_classes) 
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    dataloaders_dict = {"train": train_dataloader, "val": validate_dataloader}

    # Detect if GPU is available
    print("Device:", cfg.device_name)
    device = torch.device(cfg.device_name if torch.cuda.is_available() else "cpu")
    
    # Initialize the DeepLabV3 model
    net = initialize_model(cfg.num_classes, use_pretrained=True)
    net = net.to(device)

    # Define training parameters
    optimizer_ft = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, verbose=True, patience=5)
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare output directory
    pathlib.Path(cfg.dest_dir).mkdir(parents=True, exist_ok=True)
    print("Train...")

    # Train and evaluate
    train_model(net, cfg.num_classes, dataloaders_dict, 
                criterion, optimizer_ft, scheduler, 
                device, cfg.dest_dir, 
                num_epochs=cfg.num_epochs)



if __name__ == '__main__':
    main()