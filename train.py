import torch
import wandb
import pprint as pp
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from DataSyncer import SyncedDataLoader
from dataset.dataset import ForecastingTorchDataset
from utils.plotter import get_html_plot
from utils.loaders import load_hyperparams, load_model, load_optimizer

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Running on device: {}".format(device))


# Load hyperparameters and update wandb config
hyperparams = load_hyperparams()
pp.pprint(hyperparams, sort_dicts=False)
run = wandb.init(
    project="sensor-data-forecasting-{}".format(hyperparams["model"]))
wandb.config.update(hyperparams)

# Load synced data
sensor_data = SyncedDataLoader(
    path=hyperparams["dataset"], id="RX1", num_sensors=hyperparams["num_sensors"])
dataset = ForecastingTorchDataset(sensor_data, hyperparams["seq_len"])

# Split dataset
train_count = int(0.7 * dataset.__len__())
validation_count = int(0.2 * dataset.__len__())
test_count = dataset.__len__() - train_count - validation_count
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, validation_count, test_count)
)

# Dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)

# Model, criterion and optimizer
model = load_model(hyperparams)
criterion = torch.nn.MSELoss()
optimizer = load_optimizer(model, hyperparams)

# epoch loop
plot_period, plot_number = 1, 5
for epoch in range(1, hyperparams["epochs"]+1):

    print("█▓░ Epoch: {} ░▓█".format(epoch))

    # training loop
    train_it_losses = np.array([])
    train_sample_plots_outputs = []
    train_sample_plots_targets = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # (batch_size, seq_length, input_size)
        data = data.to(device=device, non_blocking=True)
        # (batch_size, seq_length, out_size)
        targets = targets.to(device=device, non_blocking=True)[
            :, :, 1:]  # remove piezo stick

        out = model(data)
        train_loss = criterion(out, targets)
        train_it_losses = np.append(train_it_losses, train_loss.item())

        # update
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        # bokeh plot of some batches every 10 epochs
        if epoch % plot_period == 0 and epoch != 0 and batch_idx <= plot_number:
            train_sample_plots_outputs.append(out)
            train_sample_plots_targets.append(targets)
            if batch_idx == plot_number:
                wandb.log({"sensor_plot_train": wandb.Html(get_html_plot(
                    train_sample_plots_outputs, train_sample_plots_targets)), "epoch": epoch}, commit=False)

    # validation loop
    validation_it_losses = np.array([])
    validation_sample_plots_outputs = []
    validation_sample_plots_targets = []

    for batch_idx, (data, targets) in enumerate(tqdm(validation_loader)):
        # (batch_size, seq_length, input_size)
        data = data.to(device=device, non_blocking=True)
        # (batch_size, seq_length, out_size)
        targets = targets.to(device=device, non_blocking=True)[
            :, :, 1:]  # remove piezo stick

        out = model(data)
        validation_loss = criterion(out, targets)
        validation_it_losses = np.append(
            validation_it_losses, validation_loss.item())

       # bokeh plot of some batches every 10 epochs
        if epoch % plot_period == 0 and epoch != 0 and batch_idx <= plot_number:
            validation_sample_plots_outputs.append(out)
            validation_sample_plots_targets.append(targets)
            if batch_idx == plot_number:
                wandb.log({"sensor_plot_validation": wandb.Html(get_html_plot(
                    validation_sample_plots_outputs, validation_sample_plots_targets)), "epoch": epoch}, commit=False)

    wandb.log({"train_loss": train_it_losses.mean(), "validation_loss": validation_it_losses.mean(
    ), "epoch": epoch})  # only log mean loss per epoch


run.finish()
