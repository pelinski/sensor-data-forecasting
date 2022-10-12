import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from lstm import CustomLSTM

num_epochs = 5

batch_size = 32
sequence_length = 8
input_size = 20
hidden_size = 12
        
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataset = torch.utils.data.TensorDataset(torch.rand(10*batch_size, sequence_length, input_size), torch.rand(10*batch_size, sequence_length, hidden_size))
test_dataset = torch.utils.data.TensorDataset(torch.rand(10*batch_size, sequence_length, input_size), torch.rand(10*batch_size, sequence_length, hidden_size))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

model = CustomLSTM(input_size, hidden_size).to(device=device, non_blocking=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("--- Device: {} ---".format(device))

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate (tqdm(train_loader)):
        data = data.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        
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