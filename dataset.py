import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetKERC21(Dataset):
    """ 
    Dataset 
    """ 
    def __init__(self, dataset_dir, data_type='train'):
        """
        Parameters
        ----------
        dataset_dir : str
            Path to the dataset
        data_type : str
            Data Type: train, val or test
        """
        if dataset_dir in [None, 'DATASET PATH HERE']:
            raise ValueError('Please provide the dataset path. Please set the path in config.ini')  
        
        self.data_type = data_type
        self.dataset_dir = dataset_dir
        # For training data we have labels available so lets just read Ids from labels file, 
        # which is better than iterating through all the folders
        if data_type == 'train':
            self.sample_ids = pd.read_csv(f"{dataset_dir}/{data_type}_labels.csv")['Id'].values
        else:
            self.sample_ids  = [file[:file.index('_')] for file in os.listdir(f'{dataset_dir}/{data_type}') if 'EEG' in file]
   
        self.quadrants = ['HAHV','HALV','LALV','LAHV'] # four quadrants in arousal, valence space
        self.personality_traits = pd.read_csv(f"{dataset_dir}/{data_type}_personality.csv")
        try:
            self.labels = pd.read_csv(f"{dataset_dir}/{data_type}_labels.csv")
        except FileNotFoundError: # val and test labels may not be available
            pass
        self.len = len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        eeg_ftrs = torch.from_numpy(np.load(f"{self.dataset_dir}/{self.data_type}/{sample_id}_EEG_ftrs.npy")).type(torch.FloatTensor)
        eda_ftrs = torch.from_numpy(np.load(f"{self.dataset_dir}/{self.data_type}/{sample_id}_EDA.npy")).type(torch.FloatTensor)
        bvp_ftrs = torch.from_numpy(np.load(f"{self.dataset_dir}/{self.data_type}/{sample_id}_BVP.npy")).type(torch.FloatTensor)
        personality_ftrs  = torch.from_numpy(self.personality_traits[self.personality_traits['Id']==sample_id].iloc[0].values[1:].astype(int)).type(torch.LongTensor) 
        sample = eeg_ftrs, eda_ftrs, bvp_ftrs, personality_ftrs, self.quadrants.index(self.labels[self.labels['Id']==sample_id]['label'].values[0]) if hasattr(self, 'labels') else [], sample_id
        return sample
    
    def __len__(self):
        return self.len