## SSBC2025_Segmentation
Repository for the baseline sclera segmentation model used in the SSBC competition at IJCB 2025. 

# Usage 
1. Set desired training and test configuration in [train_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/train_config.py) and [test_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/test_config.py).
2. Run [train_DeepLab.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/train_DeepLab.py) to train the model on either the synthetic PolyU or CrossEyed dataset.
3. Run [test_DeepLab.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/test_DeepLab.py) to test the performance of the model on real-world validation sets (MOBIUS, SBVPI, SMD).
