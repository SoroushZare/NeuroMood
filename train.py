import torch
from torch.utils.data import DataLoader
from model import EEGTransformer
from dataset import EEGDataset
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = EEGDataset("data/demo_eeg.npy")
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

model = EEGTransformer(input_dim=32, num_classes=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")