## SSBC2025_Segmentation
Repository for the baseline sclera segmentation model used in the SSBC competition at IJCB 2025. 

# Usage 
1. Run [train_DeepLab.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/train_DeepLab.py) to train the model on either the synthetic PolyU or CrossEyed dataset. Set the desired training configuration in [train_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/train_config.py).
2. Run [test_DeepLab.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/test_DeepLab.py) to test the performance of the model on real-world validation sets (MOBIUS, SBVPI, SMD). This saves the predicted labels and the computed segmentation metrics in the directories specified in [test_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/test_config.py).
3. Run [evaluate_predicted.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/evaluate_predicted.py) to evaluate the produced labels separately from the inference process.
