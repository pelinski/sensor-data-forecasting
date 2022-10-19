import torch
from torch.utils.data import Dataset
import numpy as np

class forecastingDataset(Dataset):
    def __init__(self, data, seq_len):
        self.__seq_len = seq_len
        
        # divide data into sequences of length sequence length
        num_sequences = int(np.ceil(data.shape[0] / self.seq_len))
        padding = int(num_sequences * self.seq_len - data.shape[0])
        
        self.__data = np.append(data, np.zeros((padding,8)), axis=0)
        self.__sequences = np.reshape(self.data, (num_sequences, self.seq_len, 8))
        
        self.__inputs = self.sequences[:-1] # (num_sequences-1, seq_length, num_sensors)
        self.__targets = self.sequences[1:] # (num_sequences-1, seq_length, num_sensors)
        
    @property
    def seq_len(self):
        return self.__seq_len

    @property
    def data(self):
        return self.__data

    @property
    def sequences(self):
        return self.__sequences
    
    @property
    def inputs(self):
        return self.__inputs

    @property    
    def targets(self):
        return self.__targets    
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.inputs[i]).float(), torch.from_numpy(self.targets[i]).float()
        
    