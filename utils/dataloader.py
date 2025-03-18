# from __future__ import print_function
# from __future__ import division
import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'Images','*.*'))
        self.label_files = []
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            label_filename_with_ext = f"{image_filename.replace('RGB', 'label')}.png"
            self.label_files.append(os.path.join(folder_path, 'Masks', label_filename_with_ext))
        
        print(f"{mode} dir:", folder_path)
        print(f"Number of {mode} samples", len(self.img_files))
        
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
        ])
        
    def __getitem__(self, index):
        image = Image.open(self.img_files[index])
        label = Image.open(self.label_files[index])

        # Apply Transforms
        image = self.transforms(image)
        label = transforms.functional.pil_to_tensor(label)
        
        if label.ndim == 4: 
            # Create a new label, where if label = [255, 0, 0], then new_label = 1 and so on
            new_label = torch.zeros(label.shape[1], label.shape[2])
            new_label[label[0] == 255] = 1
            new_label[label[1] == 255] = 2
            new_label[label[2] == 255] = 3
            new_label = new_label.to(dtype=torch.long)
        else: 
            new_label = label.to(dtype=torch.long)


        return image, new_label, self.img_files[index]

    def __len__(self):
        return len(self.img_files)
