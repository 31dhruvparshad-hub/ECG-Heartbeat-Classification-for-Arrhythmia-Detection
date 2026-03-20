# personalize.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.cnn_model import ECGCNN
from data.patient_data import load_patient
from config.settings import FINE_TUNE_EPOCHS, LEARNING_RATE


def personalize(record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load global model
    model = ECGCNN().to(device)
    model.load_state_dict(torch.load("global_model.pth", map_location=device))

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Load patient data
    X_adapt, y_adapt, X_test, y_test = load_patient(record)

    X_adapt = torch.tensor(X_adapt)
    y_adapt = torch.tensor(y_adapt)

    adapt_loader = DataLoader(TensorDataset(X_adapt, y_adapt), batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Fine-tune
    model.train()
    for epoch in range(FINE_TUNE_EPOCHS):
        total_loss = 0
        for x, y in adapt_loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Personalization Epoch {epoch+1}: Loss {total_loss:.4f}")

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = personalize("233")
    print("Test samples:", len(X_test))
