
import os
import torch
from tqdm import tqdm 
from torchvision.utils import save_image

from utils.dataloader import DataLoaderSegmentation
from utils.deeplab_model import initialize_model
import configs.predict_config as cfg 

def generate_predictions(model, dataloader, device, result_folder):
    print("Generate predictions...")
    
    # Define color map for Entire masks
    color_map = torch.tensor([
            [0, 0, 0],    # Class 0 (background or unclassified)
            [255, 0, 0],  # Class 1 -> Sclera
            [0, 255, 0],  # Class 2 -> Iris
            [0, 0, 255],  # Class 3 -> Pupil
    ], dtype=torch.uint8, device=device)
    
    # Iterate over data.
    for inputs, labels, img_names  in tqdm(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)['out']  
        _, preds = torch.max(outputs, 1)
        
        # Create a binary mask of the sclera        
        binarised = preds == 1
        background = outputs.clone()
        background[:, 1, ...] = float('-inf')
        outputs = torch.stack([background.amax(dim=1), outputs[:, 1, ...]], dim=1)

        # Create the probabilistic image
        diff = outputs[:, 1, ...] - outputs[:, 0, ...]
        min_, max_ = diff.amin(dim=(1, 2))[:, None, None], diff.amax(dim=(1, 2))[:, None, None]
        probabilistic = (diff - min_) / (max_ - min_)
        
        # Assign colors with indexing
        pred_mask = color_map[preds].permute(0, 3, 1, 2)
        pred_mask = pred_mask / 255 

        for img_index, img_name in enumerate(img_names):
            img_name = os.path.basename(img_name).replace(".jpg", ".png")
            save_image(binarised[img_index].to(torch.float), os.path.join(result_folder, "Binarised", img_name), format="PNG")
            save_image(probabilistic[img_index], os.path.join(result_folder, "Predictions", img_name),  format="PNG")
            if cfg.save_four_class_masks:
                save_image(pred_mask[img_index], os.path.join(result_folder, "Entire_masks", img_name.replace("jpg", "png")),  format="PNG")

        
def main():
    # Detect if GPU is available
    device = torch.device(cfg.device_name if torch.cuda.is_available() else "cpu")
    
    # Initialize model and load the trained weights 
    net = initialize_model(cfg.num_classes, use_pretrained=True)
    net.load_state_dict(torch.load(cfg.resume_where))
    net = net.to(device)
    net.eval() 
    
    for val_folder in cfg.val_folders:
        val_folder_path = os.path.join(cfg.root_folder, val_folder)
        validate_dataset = DataLoaderSegmentation(val_folder_path, "test", cfg.num_classes) 
        validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        result_folder = os.path.join(cfg.predictions_folder, val_folder)

        subfolders = ["Predictions", "Binarised"] 
        if cfg.save_four_class_masks: subfolders += ["Entire_masks"]
        for subfold in subfolders: #, "Entire_masks"]:
            os.makedirs(os.path.join(cfg.predictions_folder, val_folder, subfold) , exist_ok=True)
        
        # Train and evaluate
        generate_predictions(net, validate_dataloader, device, result_folder)


if __name__ == '__main__':
    main()