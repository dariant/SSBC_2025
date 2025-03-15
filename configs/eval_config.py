data_dir=  "OCULAR_DATASETS" 
# which_train_folder = "CrossEyed_RGB-NIR-Label_5000_500_300x400"
which_train_folder =  "PolyU+CrossEyed_RGB-NIR-Label_5000_500_300x400"

num_classes = 4 # sclera, iris, pupil, periocular
 
which_val_folders =  ["SBVPI_300x400", "SMD_300x400", "MOBIUS_300x400"]  
all_or_sclera = "sclera" 

# which_val_folders =  ["MOBIUS_300x400"]  
# all_or_sclera = "all"

pred_folder = "SSBC_SEG_PREDICTIONS"
results_folder = "SSBC_SEG_RESULTS_EVAL/" + which_train_folder 
