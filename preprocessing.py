import os
import sys, getopt
import numpy as np
import pandas as pd
import math
from scipy import signal
from tqdm import tqdm
from pyteap.signals.gsr import get_gsr_features
from pyteap.signals.bvp import get_bvp_features
from mne_features.univariate import compute_mean, compute_pow_freq_bands, compute_wavelet_coef_energy
from mne_features.univariate import  compute_std, compute_spect_entropy, compute_katz_fd, compute_kurtosis, compute_hjorth_complexity, compute_ptp_amp

import warnings
warnings.filterwarnings("ignore")


def bandpass_filter(data, low, high, sampling_rate):
    nyq = 0.5 * sampling_rate
    low = low / nyq
    high = high / nyq
    order = 2
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.lfilter(b, a, data)
    return filtered


def get_psd_features(eeg_data, sampling_freq):
    ftrs = []
    for channel_data in eeg_data:
        channel_psd_ftrs = []
        for _, (lo, hi) in {'theta':(3, 7),'alpha':(8, 13),'beta':(14, 29),'gamma':(30, 47)}.items():
            band_signal = bandpass_filter(channel_data, lo, hi, sampling_freq)
            window_size = 10  # Using 10s window with 50% overlap
            _, psd = signal.welch(band_signal, sampling_freq, nperseg=sampling_freq*window_size, noverlap=sampling_freq*window_size/2) 
            #ftr = math.log(np.max(psd))
            channel_psd_ftrs.append(psd)
        ch_ftrs = np.array(channel_psd_ftrs).ravel() #concatenate psd of 4 bands (theta alpha beta gamma) 
        ftrs.append(ch_ftrs)
        
    psd_ftrs = np.array(ftrs).ravel() #concatentate freatures from 14 channels (CH1(theta alpha beta gamma) - CH14(theta alpha beta gamma))
    return psd_ftrs

def get_psd_ftrs(eeg_data, sampling_freq):
    ftrs = []
    for channel_data in eeg_data:
        for _, (lo, hi) in {'theta':(3, 7),'alpha':(8, 13),'beta':(14, 29),'gamma':(30, 47)}.items():
            band_signal = bandpass_filter(channel_data, lo, hi, sampling_freq)
            window_size = 10  # Using 10s window with 50% overlap
            _, psd = signal.welch(band_signal, sampling_freq, nperseg=sampling_freq*window_size, noverlap=sampling_freq*window_size/2) 
            ftr = math.log(np.max(psd))
            ftrs.append(ftr)
    return np.array(ftrs)
    
    
def get_eeg_features(eeg_data, sampling_freq):
    mean = compute_mean(eeg_data) #14
    std = compute_std(eeg_data) #14
    power_freq  = compute_pow_freq_bands(sampling_freq, eeg_data, [4, 8, 13, 30, 40]) #56
    wavelet_coef_energy = compute_wavelet_coef_energy(eeg_data) #84
    spectral_entropy = compute_spect_entropy(sampling_freq, eeg_data) #14
    kurtosis  = compute_kurtosis(eeg_data) #14
    hjorth_complexity = compute_hjorth_complexity(eeg_data) #14
    p2p = compute_ptp_amp(eeg_data) #14
    fd = compute_katz_fd(eeg_data) #14
    psd = get_psd_ftrs(eeg_data, sampling_freq) #56
    return np.hstack((mean, std, power_freq, wavelet_coef_energy, spectral_entropy, kurtosis, hjorth_complexity, p2p, fd, psd))



def preprocess(input_dir, data_type):
    print(f'Preprocessing signals from {input_dir}/{data_type}')
    if not os.path.exists(f'{input_dir}/preprocessed/{data_type}'):
        os.makedirs(f'{input_dir}/preprocessed/{data_type}')
    print(f'{input_dir}/{data_type}')    
    for sample_dir in tqdm(os.listdir(f'{input_dir}/{data_type}')):
        
        EEG = pd.read_csv(f'{input_dir}/{data_type}/{sample_dir}/EEG_256.csv').to_numpy()
        #EEG Features
        features = get_eeg_features(EEG.T, 256) # 256Hz
            
        np.save(f'{input_dir}/preprocessed/{data_type}/{sample_dir}_EEG_ftrs.npy', features)

        #EDA features
        EDA = pd.read_csv(f'{input_dir}/{data_type}/{sample_dir}/EDA_4.csv', header=None).to_numpy().ravel()
        eda_features = get_gsr_features(EDA, 4)
        np.save(f'{input_dir}/preprocessed/{data_type}/{sample_dir}_EDA.npy', eda_features)
        
        
        BVP = pd.read_csv(f'{input_dir}/{data_type}/{sample_dir}/BVP_64.csv', header=None).to_numpy().ravel()
        bvp_features = get_bvp_features(BVP, 64)
        np.save(f'{input_dir}/preprocessed/{data_type}/{sample_dir}_BVP.npy', bvp_features)
         
    print(f'Preprocessed data saved to {input_dir}/preprocessed/{data_type}')


def main(argv):
    input_dir = ''
    data_type = ''
    try:
        opts, args = getopt.getopt(argv,"hr:t:",["rdir=","type="])
        if len(opts) == 0:
            print("Please provide root data directory and data type(train, val or test)")
            print('preprocessing.py -r <root_dir> -t <type>')
            sys.exit(2)

    except getopt.GetoptError:
        print('preprocessing.py -r <root_dir> -t <type>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('preprocessing.py -r <root_dir> -r <type>')
            sys.exit()
        elif opt in ("-r", "--rdir"):
            input_dir = arg
        elif opt in ("-t", "--type"):
            data_type = arg
            
    preprocess(input_dir, data_type)

 
if __name__ == "__main__":
   main(sys.argv[1:])