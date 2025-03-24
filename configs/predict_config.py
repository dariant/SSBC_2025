root_folder =  "SSBC_DATASETS_400x300/Evaluation_Sample"
val_folders = ["MOBIUS", "SMD+SLD", "Synthetic"]  

which_train_folder = "Synthetic_SIP" # "Mixed_Sclera" 
resume_where =  f"SSBC_SEG_MODELS/{which_train_folder.split('_')[0]}/best_trained_DeepLabV3.pth" 
num_classes = 4 if "SIP" in which_train_folder else 2   # 2 for Sclera train folder, and 4 for SIP 

device_name = "cuda:0"
batch_size = 4

predictions_folder = f"SSBC_SEG_PREDICTIONS/{which_train_folder.split('_')[0]}"
save_four_class_masks = False