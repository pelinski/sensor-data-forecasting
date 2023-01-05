import torch
import wandb
import random
import pprint as pp
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from DataSyncer import SyncedDataLoader
from dataset.dataset import ForecastingTorchDataset
from utils.plotter import get_html_plot
from utils.loaders import load_hyperparams, load_model, load_optimizer, load_scheduler
from utils.saver import save_model

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Running on device: {}".format(device))


# Load hyperparameters and update wandb config
hyperparams = load_hyperparams()
pp.pprint(hyperparams, sort_dicts=False)
run = wandb.init(
    project="sensor-data-forecasting-{}".format(hyperparams["model"]), settings=wandb.Settings(start_method="fork"), resume="allow",
    id=hyperparams["load_model_path"].split(
        "/")[-1] if hyperparams["load_model_path"] else wandb.util.generate_id())
wandb.config.update(hyperparams, allow_val_change=True)

# Load synced data
sensor_data = SyncedDataLoader(
    path=hyperparams["dataset"], id="RX1", num_sensors=hyperparams["num_sensors"])
dataset = ForecastingTorchDataset(
    sensor_data, hyperparams["seq_len"], hyperparams["n_tgt_win"])

# Split dataset
train_count = int(0.7 * dataset.__len__())
validation_count = int(0.2 * dataset.__len__())
test_count = dataset.__len__() - train_count - validation_count
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, validation_count, test_count)
)

# Dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, pin_memory=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=hyperparams["batch_size"], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=hyperparams["batch_size"], shuffle=True, pin_memory=True)

# Model, criterion and optimizer
model, epoch_init = load_model(hyperparams)
optimizer = load_optimizer(model, hyperparams)
criterion = torch.nn.L1Loss(reduction='mean')
scheduler = load_scheduler(hyperparams, optimizer)

# get windows with hits
threshold = 0.55
train_windows_with_hits = random.choices(list(set([e[0] for e in np.argwhere(
    train_dataset.dataset.inputs[:, :, 1] > threshold)])), k=hyperparams["plot_number"])  # sensor idx 1 --> accelerometer
validation_windows_with_hits = random.choices(list(set([e[0] for e in np.argwhere(
    validation_dataset.dataset.inputs[:, :, 1] > threshold)])), k=hyperparams["plot_number"])  # sensor idx 1 --> accelerometer
test_windows_with_hits = random.choices(list(set([e[0] for e in np.argwhere(
    test_dataset.dataset.inputs[:, :, 1] > threshold)])), k=hyperparams["plot_number"])  # sensor idx 1 --> accelerometer

# epoch loop
for epoch in range(epoch_init, hyperparams["epochs"]+1):

    print("█▓░ Epoch: {} ░▓█".format(epoch))

    # training loop
    train_it_losses = np.array([])

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
        optimizer.zero_grad(set_to_none=True)  # lower memory footprint
        train_loss.backward()
        optimizer.step()
        scheduler.step()

    # bokeh plot of some batches every hyperparams["save_and_plot_period"] epochs, save model
    if hyperparams["save_and_plot_period"] and epoch % hyperparams["save_and_plot_period"] == 0:
        save_model(model, optimizer, scheduler, hyperparams, epoch)
        inputs = torch.Tensor(
            train_dataset.dataset.inputs[train_windows_with_hits]).to(device=device)
        targets = torch.Tensor(
            train_dataset.dataset.targets[train_windows_with_hits]).to(device=device)
        outputs = model.predict(inputs)
        wandb.log({"sensor_plot_train": wandb.Html(get_html_plot(
            inputs[:, :, 1:], outputs, targets[:, :, 1:])), "epoch": epoch}, commit=False)

    # validation loop
    validation_it_losses = np.array([])

    for batch_idx, (data, targets) in enumerate(tqdm(validation_loader)):
        # (batch_size, seq_length, input_size)
        data = data.to(device=device, non_blocking=True)
        # (batch_size, seq_length, out_size)
        targets = targets.to(device=device, non_blocking=True)[
            :, :, 1:]  # remove piezo stick

        out = model.predict(data)  # using predict method to avoid backprop
        validation_loss = criterion(out, targets)
        validation_it_losses = np.append(
            validation_it_losses, validation_loss.item())

    # bokeh plot of some batches every hyperparams["save_and_plot_period"] epochs
    if hyperparams["save_and_plot_period"] and epoch % hyperparams["save_and_plot_period"] == 0:
        inputs = torch.Tensor(
            validation_dataset.dataset.inputs[validation_windows_with_hits]).to(device=device)
        targets = torch.Tensor(
            validation_dataset.dataset.targets[validation_windows_with_hits]).to(device=device)
        outputs = model.predict(inputs)
        wandb.log({"sensor_plot_validation": wandb.Html(get_html_plot(
            inputs[:, :, 1:], outputs, targets[:, :, 1:])), "epoch": epoch}, commit=False)

    losses = {"train_loss": train_it_losses.mean(
    ).round(8), "validation_loss": validation_it_losses.mean().round(8)}
    wandb.log({**losses, "epoch": epoch})  # only log mean loss per epoch
    pp.pprint(losses, sort_dicts=False)

run.finish()
