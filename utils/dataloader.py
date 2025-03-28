import torch
from torchvision import transforms
import os
import glob
from PIL import Image

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, num_classes=4):
        super(DataLoaderSegmentation, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.img_files = glob.glob(os.path.join(folder_path,'Images','*.*'))
        self.label_files = []
        for img_path in self.img_files:
            img_name = os.path.basename(img_path)
            self.label_files.append(os.path.join(folder_path, 'Masks', img_name))
        
        print(f"{mode} dir:", folder_path)
        print(f"Number of {mode} samples {len(self.img_files)}")
        
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
        ])
        
    def __getitem__(self, index):
        image = Image.open(self.img_files[index])
        # Apply Transforms
        image = self.transforms(image)
        
        if self.mode == "test":
            return image, self.img_files[index]

        label = Image.open(self.label_files[index])
        label = transforms.functional.pil_to_tensor(label)
        
        # if only training on sclera (binary mask)
        if self.num_classes == 2:
            new_label = torch.zeros(label.shape[1], label.shape[2])
            new_label[label[0] == 255] = 1
            new_label = new_label.to(dtype=torch.long)

        # if training on all 4 classes
        # Create a new label, where if label = [255, 0, 0], then new_label = 1 and so on
        else : 
            new_label = torch.zeros(label.shape[1], label.shape[2])
            new_label[label[0] == 255] = 1
            new_label[label[1] == 255] = 2
            new_label[label[2] == 255] = 3
            new_label = new_label.to(dtype=torch.long)
        
        return image, new_label, self.img_files[index]

    def __len__(self):
        return len(self.img_files)
