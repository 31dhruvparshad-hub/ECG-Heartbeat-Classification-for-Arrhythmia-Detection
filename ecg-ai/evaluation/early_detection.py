# early_detection.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cnn_model import ECGCNN
from models.personalize import personalize
from data.patient_data import load_patient


def predict_partial(model, X, ratio, device):
    length = X.shape[1]
    cut = int(length * ratio)

    X_partial = X.copy()
    X_partial[:, cut:] = 0  # hide future signal

    with torch.no_grad():
        preds = model(torch.tensor(X_partial).to(device)).argmax(dim=1).cpu().numpy()

    return preds


def early_detection(record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_test, y_test = load_patient(record)

    # Load models
    global_model = ECGCNN().to(device)
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device))
    global_model.eval()

    personal_model, _, _ = personalize(record)
    personal_model.eval()

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    global_acc = []
    personal_acc = []

    for r in ratios:
        g_pred = predict_partial(global_model, X_test, r, device)
        p_pred = predict_partial(personal_model, X_test, r, device)

        global_acc.append(np.mean(g_pred == y_test))
        personal_acc.append(np.mean(p_pred == y_test))

    plt.figure(figsize=(6,4))
    plt.plot(ratios, global_acc, marker='o', label="Global")
    plt.plot(ratios, personal_acc, marker='o', label="Personalized")
    plt.xlabel("Visible Heartbeat (%)")
    plt.ylabel("Accuracy")
    plt.title(f"Early Detection Patient {record}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    early_detection("233")
