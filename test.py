import torch
import unittest
import numpy as np
import pickle
import os
import tensorflow as tf
from DataSyncer import SyncedDataLoader
from model.lstm import CustomLSTM
from dataset.dataset import ForecastingTorchDataset
from model.transformer import TransformerEncoder
from utils.loaders import load_model, load_hyperparams_from_wandb
from export import export_to_tflite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_params = {
    "batch_size": 32,
    "seq_len": 10,
    "hidden_size": 12,
    "out_size": 30,
    "dropout": 0.2,
}


class testLSTM(unittest.TestCase):

    def test_shapes(self):

        batch_size = lstm_params["batch_size"]
        seq_len = lstm_params["seq_len"]
        hidden_size = lstm_params["hidden_size"]
        out_size = lstm_params["out_size"]

        x = torch.rand(batch_size, seq_len, hidden_size).to(
            device=device, non_blocking=True)
        lstm = CustomLSTM(**lstm_params).to(
            device=device, non_blocking=True)

        # hidden_seq, (h_t, c_t) = lstm.predict(x, return_states=True)

        out_seq = lstm.predict(x, return_states=True)

        self.assertEqual(out_seq.shape, (batch_size, out_size, hidden_size-1),
                         "output shape should be (batch_size, out_size, hidden_size")

        # self.assertEqual(h_t.shape, (batch_size, hidden_size),
        #                  "hidden state shape should be (batch_size, hidden_size)")

        # self.assertEqual(c_t.shape, (batch_size, hidden_size),
        #                  "cell state shape should be (batch_size, hidden_size)")


transformer_params = {
    "batch_size": 5,
    "seq_len": 4,
    "d_model": 64,
    "embedding_size_src": 12,
    "embedding_size_tgt": 9,
    "num_heads": 8,
    "dim_feedforward": 256,
    "dropout": 0.2,
    "n_tgt_win": 3,
    "num_encoder_layers": 7
}


class test_transformer(unittest.TestCase):

    def test_shapes(self):

        x = torch.rand(transformer_params["batch_size"], transformer_params["seq_len"], transformer_params["embedding_size_src"]).to(
            device=device, non_blocking=True)

        model = TransformerEncoder(out_size=transformer_params["n_tgt_win"]*transformer_params["seq_len"],
                                   **transformer_params).to(device=device, non_blocking=True)

        y = model.predict(x)

        self.assertEqual(y.shape, (transformer_params["batch_size"], transformer_params["n_tgt_win"]*transformer_params["seq_len"], transformer_params["embedding_size_tgt"]),
                         "output shape should be (batch_size, n_tgt_win*seq_len, embedding_size_tgt)")


class test_dataset(unittest.TestCase):

    def test_shapes(self):
        num_sensors = 8
        seq_len = 16
        n_target_windows = 3

        sensor_data = SyncedDataLoader(
            path="dataset/data/test/RX1", id="RX1", num_sensors=num_sensors)
        dataset = ForecastingTorchDataset(
            sensor_data, seq_len, n_target_windows)
        self.assertEqual(dataset.inputs.shape, (len(dataset.sequences)-n_target_windows, seq_len, num_sensors),
                         "input shape should be(len(dataset.sequences)-n_target_windows, seq_len, num_sensors)")

        self.assertEqual(dataset.targets.shape, (len(dataset.sequences)-n_target_windows, seq_len*n_target_windows, num_sensors),
                         "target shape should be(len(dataset.sequences)-n_target_windows, seq_len*n_target_windows, num_sensors)")


class test_export_to_tflite(unittest.TestCase):

    def test_export(self):
        
        with open("dataset/data/test/test-params.pk", 'rb') as f:
            hyperparams = pickle.load(f)
        
        model = CustomLSTM(hidden_size=hyperparams["num_sensors"], out_size=hyperparams["n_tgt_win"]*hyperparams["seq_len"], **hyperparams).to(
    device=device, non_blocking=True)
        
        checkpoint = torch.load(
            "dataset/data/test/test.model", map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        
        dummy_input = torch.randn(
        hyperparams["batch_size"], hyperparams["seq_len"], hyperparams["num_sensors"])
        
        converted_model_path = "dataset/data/test/test.tflite"

        export_to_tflite(model, dummy_input, converted_model_path)

        dummy_output = model.predict(dummy_input)

        interpreter = tf.lite.Interpreter(model_path=converted_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output_arr = interpreter.get_tensor(output_details[0]['index'])
        
        os.remove(converted_model_path)

        self.assertEqual(len(np.argwhere(output_arr-dummy_output.detach().numpy()>0.0001)),0,  msg="model output before and after conversion should be the same")


if __name__ == '__main__':
    unittest.main(verbosity=2)
    exit()
