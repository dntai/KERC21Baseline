import torch
from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, train_config):
        super(BaselineModel, self).__init__()
        self.lstm_input_dim = train_config['size_eeg_stat']
        self.hidden_dim =  train_config['lstm_hidden_dim']
        self.num_layers = train_config['lstm_num_layers']
        
        self.lstm = nn.LSTM(self.lstm_input_dim,self.hidden_dim, self.num_layers)
        
        self.personality_embedding = nn.Embedding(train_config['size_personality'], train_config['cat_personality']) # 4 categories
        personality_embedd_size  = train_config['size_personality'] * train_config['cat_personality']
        
        fusion_dim = train_config['lstm_hidden_dim']+ train_config['size_eda'] + train_config['size_bvp'] + personality_embedd_size
        self.clf = nn.Sequential(
                      nn.Linear(fusion_dim, 256),
                      nn.ReLU(),
                      nn.BatchNorm1d(256),
                      nn.Linear(256,  train_config['clf_out']),
                      nn.Sigmoid()) 
        
    def init_lstm_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, eeg, eda, bvp, personality):
        batch_size  = eeg.shape[0]
        self.init_lstm_hidden(batch_size)
        input = eeg.reshape(1, batch_size, self.lstm_input_dim) 
        eeg_lstm_out, _ = self.lstm(input)
        
        personality = self.personality_embedding(personality)
        personality = personality.view(personality.shape[0], -1)
        
        eeg_lstm_ftrs = eeg_lstm_out.view(eeg_lstm_out.shape[1], eeg_lstm_out.shape[2])
        
        x_fused = torch.cat([eeg_lstm_ftrs,  eda, bvp, personality], 1) #Just concatenated features from all modalities
        x = self.clf(x_fused)
        return x