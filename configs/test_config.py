data_dir=  "OCULAR_DATASETS" # "/home/darian/Desktop/SSBC_2025/OCULAR_DATASETS" 
# which_train_folder = "CrossEyed_RGB-NIR-Label_5000_500_300x400"
# which_train_folder = "PolyU_RGB-NIR-Label_5000_500_300x400"
which_train_folder = "PolyU+CrossEyed_RGB-NIR-Label_5000_500_300x400"

device_name = "cuda:1"
batch_size = 1
num_classes = 4 # sclera, iris, pupil, periocular
keep_feature_extract = False 
overlay_visualization = False
 
resume_where =  f"SSBC_SEG_EXPERIMENTS/{which_train_folder}/best_DeepLabV3_acc.pth" 
# which_val_folders = ["SBVPI_300x400", "SMD_300x400", "MOBIUS_300x400"]  
which_val_folders = ["MOBIUS_300x400"]  

all_or_sclera = "all" # "sclera"


predictions_folder = "SSBC_SEG_PREDICTIONS/" + which_train_folder 
results_folder = "SSBC_SEG_RESULTS/" + which_train_folder 