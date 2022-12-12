import torch
import unittest
import numpy as np
from DataSyncer import SyncedDataLoader
from models.lstm import CustomLSTM
from dataset.dataset import ForecastingTorchDataset
from models.transformer import TransformerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class testLSTM(unittest.TestCase):

    def test_shapes(self):

        batch_size = 32
        seq_length = 8
        input_size = 20
        hidden_size = 12

        x = torch.rand(batch_size, seq_length, input_size).to(
            device=device, non_blocking=True)
        lstm = CustomLSTM(input_size, hidden_size).to(
            device=device, non_blocking=True)

        hidden_seq, (h_t, c_t) = lstm(x, return_states=True)

        self.assertEqual(hidden_seq.shape, (batch_size, seq_length, hidden_size),
                         "output shape should be (batch_size, seq_length, hidden_size")

        self.assertEqual(h_t.shape, (batch_size, hidden_size),
                         "hidden state shape should be (batch_size, hidden_size)")

        self.assertEqual(c_t.shape, (batch_size, hidden_size),
                         "cell state shape should be (batch_size, hidden_size)")

    def test_parameters(self):

        input_size = 10
        hidden_size = 32

        lstm = CustomLSTM(input_size, hidden_size).to(
            device=device, non_blocking=True)

        total_params = sum(p.numel() for p in lstm.parameters())
        num_params = 4 * (input_size * hidden_size +
                          hidden_size * hidden_size + hidden_size)

        self.assertEqual(total_params, num_params,
                         "total number of parameters should be 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)")


transformer_params = {
    "batch_size": 5,
    "seq_len": 4,
    "d_model": 64,
    "embedding_size_src": 12,
    "embedding_size_tgt": 9,
    "num_heads": 8,
    "dim_feedforward": 256,
    "dropout": 0.2,
    "num_encoder_layers": 7
}


class test_transformer(unittest.TestCase):

    def test_shapes(self):

        x = torch.rand(transformer_params["batch_size"], transformer_params["seq_len"], transformer_params["embedding_size_src"]).to(
            device=device, non_blocking=True)

        model = TransformerEncoder(
            **transformer_params).to(device=device, non_blocking=True)

        y = model(x)

        self.assertEqual(y.shape, (transformer_params["batch_size"], transformer_params["seq_len"], transformer_params["embedding_size_tgt"]),
                         "output shape should be (batch_size, seq_len, embedding_size_tgt)")


class test_dataset(unittest.TestCase):

    def test_inputs_and_targets(self):
        num_sensors = 8
        seq_length = 16
        sensor_data = SyncedDataLoader(
            path="dataset/data/test/RX1", id="RX1", num_sensors=num_sensors)
        dataset = ForecastingTorchDataset(sensor_data, seq_length)

        self.assertEqual(dataset.inputs.shape, dataset.targets.shape,
                         "inputs and targets should have the same shape")

        self.assertTrue(np.any(dataset.inputs[1:]-dataset.targets[:-1] < 0.00001),
                        "inputs and targets should hold the same values but offset by one sequence")


if __name__ == '__main__':
    unittest.main(verbosity=2)
    exit()
