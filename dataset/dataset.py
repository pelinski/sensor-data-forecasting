import torch
from torch.utils.data import Dataset
import numpy as np


class ForecastingTorchDataset(Dataset):
    def __init__(self, data, seq_len, n_tgt_win=1):
        """Forecasting Torch Dataset constructor

        Args:
            data (np.array): Data loaded with the SyncedDataLoader class (from the DataSyncer package)
            seq_len (int): Desired sequence length. The data (continuous sensor stream) will be divided into sequences of this length.
            n_tgt_win (int): Number of windows in the target. The output sequence length will be equal to n_tgt_win*seq_len
        """
        self.__seq_len = seq_len
        self.__n_tgt_win = n_tgt_win

        # divide data into sequences of length sequence length
        num_sequences = int(np.ceil(data.shape[0] / self.seq_len))
        num_sensors = data.shape[1]

        padding = int(num_sequences * self.seq_len - data.shape[0])
        self.__data = np.append(data, np.zeros((padding, num_sensors)), axis=0)

        self.__sequences = np.reshape(
            self.data, (num_sequences, self.seq_len, num_sensors))

        # (num_sequences-1, seq_length, num_sensors)
        self.__inputs = self.sequences[:-self.n_tgt_win]
        self.__targets = []  # (num_sequences-1, seq_length, num_sensors)
        for idx in range(1, len(self.sequences)-self.n_tgt_win+1):
            # concatenate tgt windows in single sequence
            tgt_seq = np.concatenate(self.sequences[idx:self.n_tgt_win+idx])
            self.__targets.append(tgt_seq)

        self.__targets = np.stack(self.targets)

    @property
    def seq_len(self):
        return self.__seq_len

    @property
    def n_tgt_win(self):
        return self.__n_tgt_win

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
