# train_global.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config.settings import BATCH_SIZE, EPOCHS, LEARNING_RATE
from data.build_dataset import build_train_test
from models.cnn_model import ECGCNN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, y_train, _, _ = build_train_test()

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ECGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "global_model.pth")
    print("Global model saved!")


if __name__ == "__main__":
    train()
