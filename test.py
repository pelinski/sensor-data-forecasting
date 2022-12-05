import torch
import unittest
import numpy as np
from DataSyncer import SyncedDataLoader
from models.lstm import CustomLSTM
from dataset.dataset import ForecastingTorchDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class testLSTM(unittest.TestCase):
    
    def test_shapes(self):
                
        batch_size = 32
        sequence_length = 8
        input_size = 20
        hidden_size = 12

        x = torch.rand(batch_size, sequence_length, input_size).to(device=device, non_blocking=True)
        lstm = CustomLSTM(input_size, hidden_size).to(device=device, non_blocking=True)

        hidden_seq, (h_t, c_t) = lstm(x)
        
        self.assertEqual(hidden_seq.shape, (batch_size, sequence_length, hidden_size), "output shape should be (batch_size, sequence_length, hidden_size" )
        
        self.assertEqual(h_t.shape, (batch_size, hidden_size), "hidden state shape should be (batch_size, hidden_size)" )
        
        self.assertEqual(c_t.shape, (batch_size, hidden_size), "cell state shape should be (batch_size, hidden_size)" )

    def test_parameters(self):
        
        input_size = 10
        hidden_size = 32
        
        lstm = CustomLSTM(input_size, hidden_size).to(device=device, non_blocking=True)

        total_params = sum(p.numel() for p in lstm.parameters())
        num_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        
        self.assertEqual(total_params, num_params, "total number of parameters should be 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)" )
    
class test_dataset(unittest.TestCase):
    
    def test_inputs_and_targets(self):
        num_sensors = 8
        seq_length = 16 
        sensor_data = SyncedDataLoader(path="test-data/RX1",id="RX1",num_sensors=num_sensors)
        dataset = ForecastingTorchDataset(sensor_data, seq_length)
        
        self.assertEqual(dataset.inputs.shape,dataset.targets.shape, "inputs and targets should have the same shape")
        
        self.assertTrue(np.any(dataset.inputs[1:]-dataset.targets[:-1]<0.00001),"inputs and targets should hold the same values but offset by one sequence")
        
## TODO: test for transformers

if __name__ == '__main__':
    unittest.main(verbosity=2)
    exit()