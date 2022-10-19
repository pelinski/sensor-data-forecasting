import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
from DataSyncer import SyncedDataLoader
from lstm import CustomLSTM
from dataset import forecastingDataset

num_epochs = 5

batch_size = 32
sequence_length = 8
input_size = 8 # num sensors
hidden_size = 8 # num sensors
        
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_sensors = 8 # per Bela

sensor_data = SyncedDataLoader(path="data/synced/RX1",id="RX1",num_sensors=num_sensors)
dataset = forecastingDataset(sensor_data, sequence_length)

train_count = int(0.7 * dataset.__len__())
valid_count = int(0.2 * dataset.__len__())
test_count = dataset.__len__() - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, valid_count, test_count)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

model = CustomLSTM(input_size, hidden_size).to(device=device, non_blocking=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("--- Device: {} ---".format(device))

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate (tqdm(train_loader)):
        data = data.to(device=device, non_blocking=True) # (batch_size, seq_length, input_size)
        targets = targets.to(device=device, non_blocking=True) # (batch_size, seq_length, hidden_seq)
        
        hidden_seq, (h_t, c_t) = model(data)
        loss = criterion(hidden_seq, targets)
        
        # update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        optimizer.step()
        

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    # Set model to eval
    model.eval()
    
    error = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            hidden_seq, _= model(x)
            error.append(((hidden_seq-y)**2).mean().item())

    # Toggle model back to train
    model.train()
    return np.mean(error)


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:2f}")