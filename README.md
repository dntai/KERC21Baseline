# Baseline Model for KERC2021
This code is distributed as a reference baseline for KERC2021 Emotion Recognition Challenge. This baseline is provided as an example for handling the dataset used in the competition.

Baseline Model performs the task of classification of Emotion into 4 quadrants of Arousal-Valence space, namely,  
- HAHV: High Arousal, High Valence
- HALV: High Arousal, Low Valence
- LALV: Low Arousal, Low Valence
- LAHV: Low Arousal, High Valence

### Usage

preprocessing.py extracts the following features from EEG and EDA and BVP signals for train, val, or test.

`python preprocessing.py -r <PATH TO KERC21Dataset folder> -t train`

- EEG (294 features)
    - mean, std, power frequency, wavelet coef. energy, spectral entropy, kurtosis, hjroth_complexity, p2p, fd, psd
- EDA (5 features)
    - 5 statistical features using [PyTEAP](https://github.com/cheulyop/PyTEAP) package
- BVP
    - 17 statistical features using [PyTEAP](https://github.com/cheulyop/PyTEAP) package

main.py run baseline model.
    - Baseline was trained on a 'training' set with 690 samples and hyperparameters were selected based on validation data. However, validation data is not available to the participants. Therefore, the baseline code only includes the training part.
    

generate_submission_csv.py to generate output on validation or test set
    - generate submission CSV files.

### REQUIREMENTS
The baseline is implemented in PyTorch and depends on several external packages mentioned in `requirements.txt` 
- mne==0.23.0
- mne-features==0.1
- numpy==1.19.5
- PyTEAP==0.1.2
- pandas==1.2.3
- scikit-learn==0.24.2
- scipy==1.6.3
- torch==1.8.1+cu111
- tqdm==4.60.0


# Dataset

Dataset is divided into Training, Validation, and Test sets. However, only training data with labels and validation data is 

### Dataset Description

Dataset consists of following Modalities
- EEG recordings 
    - 14 channels
    - 256 Hz
    - 60s duration
- EDA
    - 4Hz
    - 60s duration
- BVP
    - 64Hz
    - 60s duration
- Temperature
    - 4Hz
    - 60s duration
- Personality Traits
    - 5 personality Traits scored Low, Low-Mid, Mid-High and High
