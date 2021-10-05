import numpy as np
import time
import torch
from dataset import DatasetKERC21
from torch.utils.data import DataLoader
from models import BaselineModel
import torch.optim as optim
from tqdm import tqdm
import copy
import warnings
from sklearn.metrics import f1_score
from loss import FocalLoss
warnings.filterwarnings("ignore")

class Baseline():
    ''' 
    Baseline for Classification
    
    # NOTE: Although the Hyperparameters were selected based on validation data, 
    # validation Labels are not available to the participants of KERC'21, therefore the baseline code only contains the training part.
    # Please use a part of training set for validation.
    
    '''
    def __init__(self, device, train_configs):
        super(Baseline, self).__init__()
        self.device = device
        self.train_config = train_configs
        self.model_path = f'logs/saved_model/saved_model_{int(time.time())}.pt' 
            
    
    def train(self): 
        train_dataset = DatasetKERC21(dataset_dir=self.train_config['dataset_dir'], data_type='train')
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'], shuffle=False)       
        
        model = BaselineModel(self.train_config).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.train_config['learning_rate'])
        criterion = FocalLoss()
        NUM_EPOCHS = self.train_config['epochs']
        best_loss = 100
         
        #intital model state
        best_model_wts = copy.deepcopy(model.state_dict())
        epoch_tqdm = tqdm(total=NUM_EPOCHS, desc='Epoch', position=0)
        training_info = tqdm(total=0, position=1, bar_format='{desc}')
        for epoch in range(NUM_EPOCHS): 
            model.train()
            running_loss = 0.0 
            train_label_list = []
            train_pred_list = []
                
            for eeg, eda, bvp, personality, labels, _ in train_dataloader:
                #eeg_gasf = eeg_gasf.to(self.device)
                eeg = eeg.to(self.device)
                eda = eda.to(self.device)
                bvp = bvp.to(self.device)
                personality  = personality.to(self.device)
                labels = labels.to(self.device) 
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    outputs = model(eeg, eda, bvp, personality)
                    loss = criterion(outputs, labels)
                    
                    train_label_list.extend(labels.detach().cpu())
                    train_preds = np.argmax(outputs.detach().cpu(), axis=1)
                    train_pred_list.extend(train_preds)
                    
                    loss.backward()
                    optimizer.step()  
                
                running_loss += loss.item() * eeg.size(0)
                    
            total_count =  len(train_dataset)
            epoch_loss = running_loss / total_count
            
            train_f1 = f1_score(train_label_list, train_pred_list, average='micro')
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            training_info.set_description_str(f'Epoch {epoch+1}/{NUM_EPOCHS},  Loss:{epoch_loss:.4f}, F1:{train_f1:.4f},  Best Loss:{best_loss:.4f}')
            epoch_tqdm.update(1) 
        
        #load best model weights and save
        model.load_state_dict(best_model_wts)
        torch.save(model,  f'logs/saved_model/saved_model.pt')
