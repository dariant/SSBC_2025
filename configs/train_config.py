# data_dir=  "/media/darian/Storage_0/BiOcularGAN_IJCB_EXPERIMENTS/SEG_EXPERIMENTS/OCULAR_SEG_DATASETS/OCULAR_DATASETS"
data_dir = "/home/darian/Desktop/SSBC_2025/OCULAR_DATASETS/Synthetic_datasets"

# main_folder = "CrossEyed_RGB-NIR-Label_5000_500_400x300"
main_folder = "PolyU_RGB-NIR-Label_5000_500_400x300"

which_train_folder = f"{main_folder}/training_5000"
which_val_folder = f"{main_folder}/validation_500"

dest_dir = f"SSBC_SEG_EXPERIMENTS/{main_folder}"
device_name = "cuda:0"

batch_size = 8
num_classes = 4 # sclera, iris, pupil, periocular
num_epochs = 50
keep_feature_extract = False 
w = [] # weights for each class
