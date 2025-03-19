data_dir = "SSBC_DATASETS_400x300"
main_folder = "Synthetic_SIP" # "Syn+SBVPI_Sclera" 
num_classes = 4 # Set to 4, for SIP training folder, and 2 for Sclera 

device_name = "cuda:0"
batch_size = 8
num_epochs = 50 

dest_dir = f"SSBC_SEG_EXPERIMENTS/{main_folder}"