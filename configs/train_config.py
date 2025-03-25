root_folder = "SSBC_DATASETS_400x300"
data_folder = "Synthetic_SIP" # "Mixed_Sclera"
num_classes = 4 if "SIP" in data_folder else 2  # 4 for SIP data, and 2 for only Sclera 

device_name = "cuda:0"
batch_size = 8
num_epochs = 50 

dest_folder = f"SSBC_SEG_MODELS/{data_folder.split('_')[0]}"
