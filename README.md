# SSBC 2025: Privacy-enhancing Sclera Segmentation Benchmarking Competition

<a href='https://arxiv.org/abs/2508.10737'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://ijcb2025.ieee-biometrics.org/competitions/'><img src='https://img.shields.io/badge/Competition_at-IJCB_2025-blue'></a>

This repository contains information regarding the Privacy-enhancing Sclera Segmentation Benchmarking Competition, which was held at IJCB 2025.
Included are the training and evaluation protocols and datasets, along with the starter kit code for an example baseline segmentation model. 
  
## Training datasets and protocol
The following datasets were used as part of SSBC for training sclera segmentation models: 
- [SBVPI (Sclera Blood Vessels, Periocular and Iris dataset)](https://sclera.fri.uni-lj.si/datasets.html)  
- [SynCROI (Synthetic Cross-Racial Ocular Image dataset)](https://sclera.fri.uni-lj.si/datasets.html) (To be released)

Both datasets come in two versions, with either 2-class sclera masks or 4-class (SIP - Sclera, Iris, Pupil) masks. In the 2-class masks, the white region represents the sclera, while the black region represents the background (everything else). In the 4-class masks, the classes are represented by different colors: red – sclera, green – iris, blue – pupil, black – periocular/background. SBVPI contains significantly fewer images in the 4-class version due to a lack of handcrafted annotations, while SynCROI contains the same images in both versions. The first 5000 training and first 500 validation samples of SynCROI are based on Caucasian subjects, while the rest are based on Asian subjects.

The participants were asked to train two versions of their segmentation method, with the first trained exclusively on SynCROI, while the second could use any mix of SynCROI and SBVPI data. Any other decisions regarding the model architecture and training procedure were left to the participants’ discretion, including e.g.:
- Training either a 4-class or a 2-class segmentation model,
- The training/validation set distribution split,
- Sample balancing to equalize the real/synthetic sample representation,
- Data augmentation during training

## Evaluation datasets and protocol
The following datasets were used as part of SSBC to evaluate the sclera segmentation performance of submitted models: 
- [MOBIUS (Mobile Ocular Biometrics In Unconstrained Settings dataset)](https://sclera.fri.uni-lj.si/datasets.html)
- [SMD+SLD (Sclera Mobile Dataset + Sclera Liveness Dataset)](https://sites.google.com/site/dasabhijit2048/datatsets)
- [SynMOBIUS (Synthetic MOBIUS dataset)](https://sclera.fri.uni-lj.si/datasets.html) (To be released)

The trained segmentation models should be used to predict the sclera regions in the images of the different evaluation datasets. 
The predictions should include: 
- Binarised segmentation masks, where white pixels denote sclera and black denote everything else,
- Grayscale probabilistic masks, where the intensity represents the probability of the pixel belonging to the sclera

For evaluation the masks should be saved in image form (PNG format) at a size of 400x300 (WxH).  
Do note, that the evaluation is performed on the sclera alone, however, 4-class segmentation models might better separate the sclera from other regions. 
The included generation script ([generate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/generate_predictions.py)) contains a basic approach for generating both probabilistic and binarised predictions, but different probability generation and binarization/thresholding strategy can also be used.

## Example: Segmentation model training and evaluation
1. Clone the repository and place the downloaded ocular datasets in the "SSBC_DATASETS_400x300" directory.  
2. Run [train_model.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/train_model.py) to train the model on a desired dataset (e.g. the SSBC2025 Synthetic dataset).
3. Run [generate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/generate_predictions.py) to perform segmentation with the trained model on the evaluation datasetssets and save the binarised and probabilstic predictions. 
4. Run [evaluate_predictions.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/evaluate_predictions.py) to evaluate the predictions separately in terms of F1 score and IoU for binarised masks, and Precision, Recall, and F1 score for probabilistic images. 

Set the desired training, testing, and evaluation configurations in [train_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/train_config.py), [predict_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/predict_config.py), and [eval_config.py](https://github.com/dariant/SSBC2025_Segmentation/blob/main/configs/eval_config.py).



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
