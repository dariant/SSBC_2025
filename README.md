## Sclera Segmentation and Benchmarking Competition (SSBC 2025)
Starter kit for the baseline sclera segmentation model used in the SSBC competition at IJCB 2025. 

## Usage 
1. Place the downloaded ocular datasets in the "SSBC_DATASETS_400x300" directory.  
2. Run [train_model.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/train_model.py) to train the model on a desired dataset (e.g. the SSBC2025 Synthetic dataset).
3. Run [generate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/generate_predictions.py) to perform segmentation with the trained model on real-world validation sets (e.g. MOBIUS, SMD+SLD) and save the binarised and probabilstic predictions. 
4. Run [evaluate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/evaluate_predictions.py) to evaluate the predictions separately in terms of F1 score and IoU for binarised masks, and Precision, Recall, and F1 score for probabilistic images. 

Set the desired training, testing, and evaluation configurations in [train_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/train_config.py), [predict_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/predict_config.py), and [eval_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/eval_config.py).

## Training datasets
The following datasets were used as part of SSBC for training sclera segmentation models: 
- [SBVPI (Sclera Blood Vessels, Periocular and Iris dataset)](https://sclera.fri.uni-lj.si/datasets.html)  
- [SynCROI (Synthetic Cross-Racial Ocular Image dataset)](https://sclera.fri.uni-lj.si/datasets.html) (To be released)

Both training datasets come in 2 versions. One version contains the images and the 2-class sclera masks, while the other contains the images and corresponding 4-class (SIP - Sclera, Iris, Pupil) masks. In the 2-class masks, the white region represents the sclera, while the black region represents the background (everything else). In the 4-class masks, the classes are represented by different colors: red – sclera, green – iris, blue – pupil, black – periocular/background. The SSBC Synthetic dataset contains the same image samples in both versions (thus only differing in the masks), however, SBVPI contains significantly fewer images in the 4-class version due to a lack of handcrafted annotations. The first 5000 training and first 500 validation samples of the SSBC Synthetic dataset are based on Caucasian subjects, while the rest are based on Asian subjects.

## Training protocol of SSBC

The participants were asked to train two versions of their segmentation method, with the trained exclusively on SynMOBIUS, while the second could use any mix of SynMOBIUS and SBVPI data. Any other decisions regarding the model architecture and training procedure were left to the participants’ discretion, including:
- Training either a 4-class or a 2-class segmentation model,
- The training/validation set distribution split,
- Sample balancing to equalize the real/synthetic sample representation,
- Data augmentation during training,
- Training image size, however, the final evaluation was performed on 400x300 images


## Evaluation datasets 
The following datasets were used as part of SSBC for evaluating sclera segmentation performance: 
- [MOBIUS (Mobile Ocular Biometrics In Unconstrained Settings dataset)](https://sclera.fri.uni-lj.si/datasets.html)
- [SMD+SLD (Sclera Mobile Dataset + Sclera Liveness Dataset)](https://sites.google.com/site/dasabhijit2048/datatsets)
- [SynMOBIUS (Synthetic MOBIUS dataset)](https://sclera.fri.uni-lj.si/datasets.html) (To be released)

The evaluation will be performed on the sclera alone, however, 4-class models might better separate the sclera from other regions.



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
