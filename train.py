import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split

from DataSyncer import SyncedDataLoader
from lstm import CustomLSTM
from dataset import forecastingDataset
from plotter import get_html_plot



run = wandb.init()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Running on device: {}".format(device))

# Hyperparameters either come from wandb or are default values
hyperparams ={"epochs": wandb.config.epochs if wandb.config.__contains__("epochs") else 200, 
                  "batch_size": wandb.config.batch_size if wandb.config.__contains__("batch_size") else 32, 
                  "sequence_length": wandb.config.sequence_length if wandb.config.__contains__("sequence_length") else 16,
                  "learning_rate": wandb.config.learning_rate if wandb.config.__contains__("learning_rate") else 0.001, 
                  "optimizer":wandb.config.optimizer if wandb.config.__contains__("optimizer") else "adam"}
        

# Load synced data
num_sensors = 8 # per Bela
sensor_data = SyncedDataLoader(path="data/synced/RX1",id="RX1",num_sensors=num_sensors)
dataset = forecastingDataset(sensor_data, hyperparams["sequence_length"])
# Split dataset
train_count = int(0.7 * dataset.__len__())
validation_count = int(0.2 * dataset.__len__())
test_count = dataset.__len__() - train_count - validation_count
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, validation_count, test_count)
)
# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)

# Model, criterion and optimizer
model = CustomLSTM(input_size=num_sensors, hidden_size=num_sensors).to(device=device, non_blocking=True)
criterion = torch.nn.MSELoss()
if hyperparams["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
elif hyperparams["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["learning_rate"])
else:
    optimizer = None

# epoch loop
plot_period = 1
plot_number = 5
for epoch in range(hyperparams["epochs"]):
    
    print("| Epoch: {} |".format(epoch))
    
    # training loop
    train_it_losses = np.array([])
    train_sample_plots_outputs = []
    train_sample_plots_targets = []
    
    for batch_idx, (data, targets) in enumerate (tqdm(train_loader)):
        data = data.to(device=device, non_blocking=True) # (batch_size, seq_length, input_size)
        targets = targets.to(device=device, non_blocking=True) # (batch_size, seq_length, hidden_seq)
        
        hidden_seq, (h_t, c_t) = model(data)
        train_loss = criterion(hidden_seq, targets)
        train_it_losses = np.append(train_it_losses, train_loss.item())  
              
        # update
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step() 
        
        # bokeh plot of some batches every 10 epochs
        if epoch%plot_period == 0 and epoch!=0 and batch_idx <= plot_number:
            train_sample_plots_outputs.append(hidden_seq)
            train_sample_plots_targets.append(targets)
            if batch_idx == plot_number:
                wandb.log({"sensor_plot_train":wandb.Html(get_html_plot(train_sample_plots_outputs, train_sample_plots_targets)), "epoch":epoch}, commit=False)
        
        
    # validation loop
    validation_it_losses = np.array([])
    validation_sample_plots_outputs = []
    validation_sample_plots_targets = []
    
    for batch_idx, (data, targets) in enumerate (tqdm(validation_loader)):
        data = data.to(device=device, non_blocking=True) # (batch_size, seq_length, input_size)
        targets = targets.to(device=device, non_blocking=True) # (batch_size, seq_length, hidden_seq)
        
        hidden_seq, (h_t, c_t) = model(data)
        validation_loss = criterion(hidden_seq, targets)
        validation_it_losses = np.append(validation_it_losses,validation_loss.item())
        
       # bokeh plot of some batches every 10 epochs
        if epoch%plot_period == 0 and epoch!=0 and batch_idx <= plot_number:
            validation_sample_plots_outputs.append(hidden_seq)
            validation_sample_plots_targets.append(targets)
            if batch_idx == plot_number:
                wandb.log({"sensor_plot_validation":wandb.Html(get_html_plot(validation_sample_plots_outputs, validation_sample_plots_targets)), "epoch":epoch}, commit=False)
        
    
    wandb.log({"train_loss": train_it_losses.mean(), "validation_loss": validation_it_losses.mean(), "epoch": epoch}) # only log mean loss per epoch
        

run.finish()