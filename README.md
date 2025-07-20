## Sclera Segmentation and Benchmarking Competition (SSBC 2025)
Starter kit for the baseline sclera segmentation model used in the SSBC competition at IJCB 2025. 

## Usage 
1. Place the downloaded ocular datasets in the "SSBC_DATASETS_400x300" directory.  
2. Run [train_model.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/train_model.py) to train the model on a desired dataset (e.g. the SSBC2025 Synthetic dataset).
3. Run [generate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/generate_predictions.py) to perform segmentation with the trained model on real-world validation sets (e.g. MOBIUS, SMD+SLD) and save the binarised and probabilstic predictions. 
4. Run [evaluate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/evaluate_predictions.py) to evaluate the predictions separately in terms of F1 score and IoU for binarised masks, and Precision, Recall, and F1 score for probabilistic images. 

Set the desired training, testing, and evaluation configurations in [train_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/train_config.py), [predict_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/predict_config.py), and [eval_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/eval_config.py).

## Datasets
The following real-world and synthetic datasets were used as part of SSBC: 
- [SBVPI](https://sclera.fri.uni-lj.si/datasets.html) for training segmentation models
- SynCROI for training segmentation models (To be released)
- [MOBIUS](https://sclera.fri.uni-lj.si/datasets.html) for evaluating segmentation performance
- [SMD+SLD](https://sites.google.com/site/dasabhijit2048/datatsets), for evaluating segmentation performance
- SynMOBIUS for evaluating segmentation performance (To be released)


## Requirements & Installation
```bash
conda create -n ssbc2025 python=3.10
conda activate ssbc2025
pip install -r requirements.txt
```

## Citation
If you use the code or results of this repository, please cite the SSBC paper:
```
TBA
```



## Acknowledgements

Supported in parts by the Slovenian Research and Innovation Agency (ARIS) through the Research Programmes P2-0250 (B) "Metrology and Biometric Systems" and P2--0214 (A) “Computer Vision”, the ARIS Project J2-50065 "DeepFake DAD" and the ARIS Young Researcher Programme.
