data_dir =  "SSBC_DATASETS_400x300/Evaluation_Sample"
which_val_folders = ["MOBIUS", "SMD+SLD"]  

which_train_folder = "Synthetic_SIP"
num_classes = 4 # Set to 4, for SIP training folder, and 2 for Sclera 
resume_where =  f"SSBC_SEG_EXPERIMENTS/{which_train_folder}/best_trained_DeepLabV3.pth" 

device_name = "cuda:0"
batch_size = 4

predictions_folder = f"SSBC_SEG_PREDICTIONS/{which_train_folder}"
save_four_class_masks = False