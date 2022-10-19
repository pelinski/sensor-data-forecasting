import torch
from torch.utils.data import Dataset
import numpy as np

class forecastingDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        
        # divide data into sequences of length sequence length
        num_sequences = int(np.ceil(data.shape[0] / self.seq_len))
        padding = int(num_sequences * self.seq_len - data.shape[0])
        
        self.data = np.append(data, np.zeros((padding,8)), axis=0)
        self.sequences = np.reshape(self.data, (num_sequences, self.seq_len, 8))
        
        self.inputs = self.sequences[:-1] # (num_sequences-1, seq_length, num_sensors)
        self.targets = self.sequences[1:] # (num_sequences-1, seq_length, num_sensors)
        
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.inputs[i]).float(), torch.from_numpy(self.targets[i]).float()
        
    