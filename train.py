import torch
import random
import pprint as pp
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from tinynn.converter import TFLiteConverter


from DataSyncer import SyncedDataLoader
from dataset.dataset import ForecastingTorchDataset
from lstm import CustomLSTM

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Running on device: {}".format(device))

# <-- Define training and model parameters -->
hyperparams = {
    "model": "lstm",
    # "dataset": "dataset/data/chaos-bells-4/processed/RX0",
    "dataset": "dataset/data/test/RX0",
    "num_sensors": 2,
    "hidden_size": 32,
    "epochs": 2,
    "learning_rate": 0.001,
    "seq_len": 10,
    "n_tgt_win": 2,
    "batch_size": 64,
    "dropout": 0.2}

# <-- Dataset loading -->
sensor_data = SyncedDataLoader(
    path=hyperparams["dataset"], id="RX1", num_sensors=hyperparams["num_sensors"])
dataset = ForecastingTorchDataset(
    sensor_data, hyperparams["seq_len"], hyperparams["n_tgt_win"])

# Split dataset
train_count = int(0.7 * dataset.__len__())
test_count = dataset.__len__() - train_count
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, test_count)
)

# Dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=hyperparams["batch_size"], shuffle=True, pin_memory=True)

# <-- Model, criterion and optimizer -->
model = CustomLSTM(input_size=hyperparams["num_sensors"], out_size=hyperparams["n_tgt_win"]
                   * hyperparams["seq_len"], **hyperparams).to(device=device, non_blocking=True)
optimizer = torch.optim.Adam(
    model.parameters(), lr=hyperparams["learning_rate"])
criterion = torch.nn.MSELoss(reduction='mean')

# <-- Training loop -->
for epoch in range(1, hyperparams["epochs"]+1):

    print("█▓░ Epoch: {} ░▓█".format(epoch))

    # training loop
    train_it_losses = np.array([])

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # (batch_size, seq_length, input_size)
        data = data.to(device=device, non_blocking=True)
        # (batch_size, seq_length, out_size)
        targets = targets.to(device=device, non_blocking=True)[
            :, :, 1:]  # remove piezo stick

        # update
        optimizer.zero_grad(set_to_none=True)  # lower memory footprint
        out = model(data)
        train_loss = torch.sqrt(criterion(out, targets))
        train_it_losses = np.append(train_it_losses, train_loss.item())
        train_loss.backward()
        optimizer.step()

    # test loop
    test_it_losses = np.array([])

    for batch_idx, (data, targets) in enumerate(tqdm(test_loader)):
        # (batch_size, seq_length, input_size)
        data = data.to(device=device, non_blocking=True)
        # (batch_size, seq_length, out_size)
        targets = targets.to(device=device, non_blocking=True)[
            :, :, 1:]  # remove piezo stick

        out = model.predict(data)  # using predict method to avoid backprop
        test_loss = torch.sqrt(criterion(out, targets))
        test_it_losses = np.append(
            test_it_losses, test_loss.item())

    losses = {"train_loss": train_it_losses.mean().round(
        8), "test_loss": test_it_losses.mean().round(8)}
    pp.pprint(losses, sort_dicts=False)

    # save model every 10 epochs and at last epoch
    if (epoch % 10 == 0 or epoch == hyperparams["epochs"]):
        save_filename = f"epoch_{epoch}.model"
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
        ), 'optimizer_state_dict': optimizer.state_dict(), 'hyperparams': hyperparams}, save_filename)


# <-- Export and save model -->
def export_to_tflite(model, dummy_input, converted_model_path):
    with torch.no_grad():
        model.cpu()
        model.eval()
        converter = TFLiteConverter(model, dummy_input, converted_model_path)
        converter.convert()


# dummy input
dummy_input = torch.randn(
    hyperparams["batch_size"], hyperparams["seq_len"], hyperparams["num_sensors"])

# convert to tflite
export_to_tflite(model, dummy_input, f"epoch_{hyperparams['epochs']}.tflite")
